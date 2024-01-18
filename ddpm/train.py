import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Visualization tools
from torchinfo import summary
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from model.model import Unet
from utils.utils import (
    show_images, load_fashionMNIST, load_transformed_fasionMNIST, get_loss, plot_sample, show_tensor_image
)

NUM_CLASSES = 10
IMG_SIZE = 16
IMG_CH = 1
BATCH_SIZE = 128

train_set = torchvision.datasets.FashionMNIST(
    "./data/", download=True, transform=transforms.Compose([transforms.ToTensor()])
)

plt.figure(figsize=(64, 4))
show_images(train_set)

data = load_transformed_fasionMNIST(IMG_SIZE)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda: ", torch.cuda.is_available())

model = Unet(IMG_CH, IMG_SIZE).to(device)

print("Num params: ", sum(p.numel() for p in model.parameters()))

# print(summary(model, (BATCH_SIZE, IMG_CH, IMG_SIZE, IMG_SIZE)))

# training
optimizer = Adam(model.parameters(), lr=0.0001)
epochs = 2

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        images = batch[0].to(device)
        loss = get_loss(model, images)
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0 and step % 100 == 0:
            print(f"Epoch {epoch} | Step {step:03d} Loss: {loss.item()}")
            plot_sample(images, model)
            plt.savefig(f'image/train/{epoch}_{step}.png')

# Validation
model.eval()
for _ in range(10):
    noise = torch.randn((1, IMG_CH, IMG_SIZE, IMG_SIZE), device=device)
    result = model(noise)
    nrows = 1
    ncols = 2
    samples = {
        "Noise" : noise,
        "Generated Image" : result
    }
    for i, (title, img) in enumerate(samples.items()):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_title(title)
        show_tensor_image(img)
    plt.show()
    plt.savefig(f'image/test/{i}.png')

