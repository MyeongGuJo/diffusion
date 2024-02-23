from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        kernel_size = 3
        stride = 1
        padding = 1
        
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        kernel_size = 3
        stride = 1
        padding = 1
        
        strideT = 2
        out_paddingT = 1
        
        super().__init__()
        layers = [
            nn.ConvTranspose2d(2 * in_ch, out_ch, kernel_size, strideT, padding, out_paddingT),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedBlock(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim, 1, 1)),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        input = input.view(-1, self.input_dim)
        return self.model(input)


class Unet(nn.Module):
    def __init__(self, ch, size, timestep=1000):
        super().__init__()
        down_chs = (16, 32, 64)
        up_chs = down_chs[::-1]
        latent_image_size = size // 4
        t_dim = 1
        
        self.in_shape = (1, ch, size, size)
        
        self.down0 = nn.Sequential(
            nn.Conv2d(ch, down_chs[0], 3, padding=1),
            nn.BatchNorm2d(down_chs[0]),
            nn.ReLU(),
        )
        
        self.down1 = DownBlock(down_chs[0], down_chs[1])
        self.down2 = DownBlock(down_chs[1], down_chs[2])
        self.to_vec = nn.Sequential(nn.Flatten(), nn.ReLU())
        
        self.dense_emb = nn.Sequential(
            nn.Linear(down_chs[2]*latent_image_size**2, down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[2]*latent_image_size**2),
            nn.ReLU(),
        )
        self.temb_1 = EmbedBlock(t_dim, up_chs[0])
        self.temb_2 = EmbedBlock(t_dim, up_chs[1])
        
        self.up0 = nn.Sequential(
            nn.Unflatten(1, (up_chs[0], latent_image_size, latent_image_size)),
            nn.Conv2d(up_chs[0], up_chs[0], 3, padding=1),
            nn.BatchNorm2d(up_chs[0]),
            nn.ReLU(),
        )
        self.up1 = UpBlock(up_chs[0], up_chs[1])
        self.up2 = UpBlock(up_chs[1], up_chs[2])
        
        self.out = nn.Sequential(
            nn.Conv2d(up_chs[-1], up_chs[-1], 3, 1, 1),
            nn.BatchNorm2d(up_chs[-1]),
            nn.ReLU(),
            nn.Conv2d(up_chs[-1], ch, 3, 1, 1),
        )
        
        self.timestep = timestep
        self.betas = torch.linspace(1e-4, 2e-2, self.timestep)
    
    def forward(self, x, t):
        timestep = torch.tensor([self.timestep], device=x.device)
        
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        latent_vec = self.to_vec(down2)
        
        t = t.float() / timestep
        latent_vec = self.dense_emb(latent_vec)
        temb_1 = self.temb_1(t)
        temb_2 = self.temb_2(t)
        
        up0 = self.up0(latent_vec)
        up1 = self.up1(up0+temb_1, down2)
        up2 = self.up2(up1+temb_2, down1)
        return self.out(up2)

    def get_loss(self, input, t):
        betas = self.betas.to(input.device)
        
        alphas = 1 - betas
        alphas_bar = alphas[:t].prod()
        
        noise = torch.randn_like(input)
        input = alphas_bar.sqrt() * input + (1 - alphas_bar).sqrt() * noise
        
        pred = self(input, t)
        
        #loss = F.mse_loss(pred, noise)
        loss = (noise - pred).square().mean()
        
        return loss

    def sampling(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        betas = self.betas.to(device)
        
        alphas = 1 - betas
        
        in_dim = self.in_shape
        x = torch.randn(in_dim, device=device)
        
        iteration = tqdm(range(0, self.timestep)[::-1])
        iteration.set_description('Sampling...')
        
        for t in iteration:
            alphas_bar = alphas[:t].prod()
            sigma = betas[t].sqrt()
            
            if t > 1:
                z = torch.randn(in_dim, device=device)
            else:
                z = 0
                
            pred = self(x, torch.tensor([t], device=x.device))
            x = (1 / alphas[t]) * (x - (1 - alphas[t]) / (1 - alphas_bar).sqrt() * pred) + sigma * z
        
        return x
