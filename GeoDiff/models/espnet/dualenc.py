import torch
from torch import nn
from tqdm.auto import tqdm
from torch_scatter import scatter_mean

from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance, eq_transform

def get_beta_schedule(
    beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps,
):
    def sigmoid(x):
        return 1 / (torch.exp(-x) + 1)
    
    if beta_schedule == 'quad':
        betas = torch.linspace(
            beta_start ** 0.5,
            beta_end ** 0.5,
            num_diffusion_timesteps,
            dtype=torch.float,
        ) ** 2
    elif beta_schedule == 'linear':
        betas = torch.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float,
        )
    elif beta_schedule == 'const':
        betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float)
    elif beta_schedule == 'jsd':    # 1/T, 1/(T-1), ..., 1
        betas = torch.Tensor([1.0]) / torch.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=torch.float,
        )
    elif beta_schedule == 'sigmoid':
        betas = torch.linspace(-6, 6, num_diffusion_timesteps, dtype=torch.float)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    
    assert betas.size(-1) == num_diffusion_timesteps
    return betas

class DualEncoderEpsNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        
        self.encoder_global = SchNetEncoder(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=config.cutoff,
            smooth=config.smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=config.hidden_dim,
            num_convs=config.num_convs_local,
        )
        
        """(
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act,
        )
        
        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation=config.mlp_act,
        )
        
        self.model_type = config.type     # diffusion or dsm(denoising score match)
        
        if self.model_type == 'diffusion':
            # denoising deffusion
            # betas
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            
            # variances
            self.alphas = nn.Parameter(torch.tensor([1.]) - betas, requires_grad=False).cumprod(dim=0)
            self.num_timesteps = self.betas.size(0)
        elif self.model_type == 'dsm':
            pass
        
    
    def forward(self, atom_type, pos, bond_index, bond_type, batch, time_step, 
                edge_index=None, edge_type=None, edge_length=None, return_edges=False, 
                extend_order=True, extend_radius=True, is_sidechain=None):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.config.edge_order,
                cutoff=self.config.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
        local_edge_mask = is_local_edge(edge_type)  # (E, ), edge_type이 0이면 masking
        
        if self.model_type == 'diffusion':
            sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device) # (E, 1)
        
        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type,
        )   # Embed edges
        
        # Global
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )
        
        # Assemble pair-wise feature
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )   # (E_global, 2H)
        
        # Invariant features of edges (radius graph, global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge) # (E_global, 1)
        
        # Encoding local
        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type,
        )   # Embed edges
        
        # Local
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        
        # assemble pairwise features
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )   # (E_local, 2H)
        
        # invariant features of edges (bond_graph, local)
        if isinstance(sigma_edge, torch.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask])
            # (E_local, 1)
        else:
            edge_inv_local = self.grad_local_dist_mlp[h_pair_local] * (1.0 / sigma_edge)
            # (E_local, 1)
        
        if return_edges:
            return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask
        else:
            return edge_inv_global, edge_inv_local
        
    def get_loss(
        self, atom_type, pos, bond_index, bond_type, batch, num_node_per_graph, num_graphs,
        anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False,
        extend_order=True, extend_radius=True, is_sidechain=None,
    ):
        if self.model_type == 'diffusion':
            return self.get_loss_diffusion(
                atom_type, pos, bond_index, bond_type, batch, num_graphs, anneal_power,
                return_unreduced_loss, return_unreduced_edge_loss,
                extend_order, extend_radius, is_sidechain,
            )
        else:
            pass
    
    def get_loss_diffusion(
        self, atom_type, pos, bond_index, bond_type, batch, num_graphs, anneal_powar=2.0,
        return_unreduced_loss=False, return_unreduced_edge_loss=False,
        extend_order=True, extend_radius=True, is_sidechain=None,
    ):
        N = atom_type.size(0)
        node2graph = batch
        
        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
        # Sample noise levels
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2 + 1, ), device=pos.device,
        )
        time_step = torch.cat(
            [time_step, self.num_timesteps-time_step-1], dim=0,
        )[:num_graphs]
        a = self.alphas.index_select(0, time_step)  # (G, )
        # Perturb pos
        a_pos = a.index_select(0, node2graph).unsqueeze(-1) # (N, 1)
        
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        # Update invariant edge features, as shown in equation 5-7
        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
            atom_type = atom_type,
            pos = pos_perturbed,
            bond_index = bond_index,
            bond_type = bond_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius,
            is_sidechain = is_sidechain,
        )   # (E_global, 1), (E_local, 1)
        
        edge2graph = node2graph.index_select(0, edge_index[0])
        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)    # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length
        # Filtering for protein
        train_edge_mask = is_train_edge(edge_index, is_sidechain)   # all True. (is_sidechain == None)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        if self.config.edge_encoder == 'gaussian':
            pass    # my edge_encoder is 'mlp'
        d_target =(d_gt - d_perturbed) / (1.0 - a_edge).sqrt() *a_edge.sqrt()  # (E_global, 1), denoising direction

        global_mask = torch.logical_and(torch.logical_or(d_perturbed <= self.config.cutoff, local_edge_mask.unsqueeze(-1)),
                                        ~local_edge_mask.unsqueeze(-1)
                        )
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global) ** 2
        loss_global = 2 * torch.sum(loss_global, dim=-1, keepdim=True)

        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:local_edge_mask], edge_length[:local_edge_mask])
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        loss_local = (node_eq_local - target_pos_local) ** 2
        loss_local = 5 * torch.sum(loss_local, dim=-1, keepdim=True)

        # loss for atomic eps regression
        loss = loss_global + loss_local

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss

    def langevin_dynamics_sample(
            self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius=True,
            n_steps=100, step_lr=1e-6, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
            global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs
    ):
        if self.model_type == 'diffusion':
            return self.langevin_dynamics_sample_diffusion(
                atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius,
                n_steps, step_lr, clip, clip_local, clip_pos, min_sigma, is_sidechain,
                global_start_sigma, w_global, w_reg,
                sampling_type=kwargs.get("sampling_type", 'ddpm_noisy'), eta=kwargs.get("eta", 1.)
            )
        elif self.model_type == 'dsm':
            pass

    def langevin_dynamics_sample_diffusion(
            self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs, extend_order, extend_radius=True,
            n_steps=100, step_lr=1e-6, clip=1000, clip_local=None, clip_pos=None, min_sigma=0, is_sidechain=None,
            global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs
    ):
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)    # view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        assert is_sidechain is None, 'sidechain prediction is not implemented'
        with torch.no_grad():
            # to test sampling with less intermediate diffusion step
            # n_steps: the num of steps
            seq = range(self.num_timesteps - n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            pos = pos_init * sigmas[-1]
            if is_sidechain is not None:
                pass
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    return_edges=True,
                    extend_order=extend_order,
                    extend_radius=extend_radius,
                    is_sidechain=is_sidechain
                )   # (E_global, 1), (E_local, 1)

                # local
                node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                # Global
                if sigmas[i] < global_start_sigma:
                    edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1),float())
                    node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0

                # Sum
                eps_pos = node_eq_local + node_eq_global * w_global # + eps_pos_reg * w_reg

                # Update

                sampling_type = kwargs.get("sampling_type", "ddpm_noisy")

                noise = torch.rand_like(pos)
                if sampling_type == "generalized" or sampling_type == "ddpm_noisy":
                    b = self.betas
                    t = t[0]
                    next_t = (torch.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    if sampling_type == "generalized":
                        eta = kwargs.get("eta", 1.)
                        et = -eps_pos
                        ## original
                        # pos0_t = (pos - et * (1 - at).sqrt()) / at.sqrt()
                        ## reweighted
                        # pos0_t = pos - et * (1 - at).sqrt() / at.sqrt()
                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        # pos_next = at_next.sqrt() * pos0_t + c1 * noise + c2 * et
                        # pos_next = pos0_t + c1 * noise / at_next.sqrt() + c2 * et / at_next.sqrt()

                        # pos_next = pos + et * (c2 / at_next.sqrt() - (1 - at).sqrt() / at.sqrt()) + noise * c1 / at_next.sqrt()
                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 5 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld<step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 3 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld<step_size_noise_generalized else step_size_noise_generalized

                        pos_next = pos - et * step_size_pos +  noise * step_size_noise

                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * pos
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size*2)

                pos = pos_next

                if is_sidechain is not None:
                    pass

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())

        return pos, pos_traj

def is_local_edge(edge_type):
    return edge_type > 0

def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center