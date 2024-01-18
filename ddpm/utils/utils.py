import torchvision
import matplotlib.pyplot as plt

def show_images(dataset, num_samples=10):
    for i, img in enumerate(dataset):
        if i == num_samples:
            return
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(torch.squeeze(img[0]))

def load_fashionMNIST(data_transform, train=TRUE):
    return torchvision.dataset.FashionMNIST(
        "./",
        download=True,
        train=train,
        transform=data_transform,
    )