from tqdm import tqdm

import torch
import diffusers

class Unet(diffusers.UNet2DModel):
    def __init__(self, ch, size, down_chs, down_block_types, up_block_types, timestep=1000):
        super().__init__(
            sample_size=size,
            in_channels=ch,
            out_channels=ch,
            layers_per_block=2,
            block_out_channels=down_chs,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        
        self.timestep = timestep
        self.betas = torch.linspace(1e-4, 2e-2, self.timestep)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, -1)
    
    def get_loss(self, input, t):
        alphas_bar = self.alphas_bar.to(input.device)
        alphas_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(input)
        input = alphas_bar_t.sqrt() * input + (1 - alphas_bar_t).sqrt() * noise
        
        pred = self(input, timestep=t)['sample']
        
        #loss = F.mse_loss(pred, noise)
        loss = (noise - pred).square().mean()
        
        return loss

    def sampling(self, size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        betas = self.betas.to(device)
        alphas = self.alphas.to(device)
        alphas_bar = self.alphas_bar.to(device)
        
        x = torch.randn(size, device=device)
        
        iteration = tqdm(range(0, self.timestep)[::-1])
        iteration.set_description('Sampling...')
        
        for t in iteration:
            sigma = betas[t].sqrt()
            
            if t > 0:
                z = torch.randn(size, device=device)
            else:
                z = 0
                
            pred = self(x, timestep=torch.tensor([t], device=x.device))['sample']
            x = (1 / alphas[t].sqrt()) * (x - (1 - alphas[t]) / (1 - alphas_bar[t]).sqrt() * pred) + sigma * z
        
        return x
