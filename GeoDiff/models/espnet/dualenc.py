import torch
from torch import nn
from torch_geometric.data import Data

from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_graph_order_radius
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder
from ..geometry import get_distance

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
        self, atom_type, pos, bond_index, bond_type, batch, num_graphs,
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

def is_local_edge(edge_type):
    return edge_type > 0