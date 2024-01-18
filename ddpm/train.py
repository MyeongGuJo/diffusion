import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Visualization tools
import matplotlib.pyplot as plt
from IPython.display import Image

# User defined libraries
from utils import utils
from model.model import Unet

IMG_SIZE = 16
IMG_CH = 1
BATCH_SIZE = 128
data, dataloader = utils.load_transformed_fasionMNIST(IMG_SIZE, BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("cuda: ", torch.cuda.is_available())

nrows = 10
ncols = 15

T = nrows * ncols
start = 0.0001
end = 0.02
B = torch.linspace(start, end, T).to(device)
a = 1. - B
a_bar = torch.cumprod(a, dim=0)
sqrt_a_bar = torch.sqrt(a_bar)
sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar)
sqrt_a_inv = torch.sqrt(1/a)
pred_noise_coeff = (1 - a) / torch.sqrt(1 - a_bar)

plt.figure(figsize=(8, 8))
x_0 = data[0][0].to(device)
x_t = x_0 # recrusion
xs = [] # Store x_t for each T to see change

model = Unet(IMG_CH, IMG_SIZE).to(device)
print("Num prams: ", sum(p.numel() for p in model.parameters()))

# training
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 3
ncols = 15

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        t = torch.randint(0, T, (BATCH_SIZE,), device=device)
        x = batch[0].to(device)
        loss = utils.get_loss(model, x, t, sqrt_a_bar, sqrt_one_minus_a_bar)
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0 and step % 100 == 0:
            print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item()}")
            sample_image(ncols, IMG_CH, IMG_SIZE, device)
            plt.savefig(f'image/train/{epoch}_{step}.png')

# final image
sample_images(ncols, IMG_CH, IMG_SIZE, device)
plt.savefig(f'image/train/final.png')

# test
model.eval()
figsize = (16, 16)
ncols = 3
for i in range(10):
    sample_images(ncols, IMG_CH, IMG_SIZE, device, figsize)
    plt.savefig(f'image/test/{i}.png')