from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

MNIST_PATH = '/Users/kenton/data/'

to_tensor = T.ToTensor()

def patchify(X: torch.Tensor, P: int) -> torch.Tensor:
    """
    Splits a single-channel image into non-overlapping P×P patches.

    Args:
        X (torch.Tensor): Input image of shape (C, H, W)
        P (int): Patch side length

    Returns:
        torch.Tensor: Tensor of shape (N_patches, P*P)
    """
    C, H, W = X.shape
    assert C == 1, "Input image must be single-channel"
    X = X.squeeze()
    assert H % P == 0 and W % P == 0, "Image dimensions must be divisible by patch size"

    # Reshape and permute to get non-overlapping patches
    patches = X.unfold(0, P, P).unfold(1, P, P)  # shape: (H/P, W/P, P, P)
    patches = patches.contiguous().view(-1, P*P)  # Flatten each patch

    return patches

def make_mnist_dataset(patch = False, patch_size = None):

    if patch:
        assert patch_size is not None
        patch_mnist = partial(patchify, P=patch_size)
        mnist_transforms = T.Compose([
        T.ToTensor(),
        patch_mnist
    ])
    else: 
        mnist_transforms = to_tensor
        
    train_ds = torchvision.datasets.MNIST(
        MNIST_PATH, train=True, download=True, transform=mnist_transforms,
    )

    val_ds = torchvision.datasets.MNIST(
        MNIST_PATH, train=False, download=True, transform=mnist_transforms,
    )

    return train_ds, val_ds

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True)
    X, y = dataset[0]
    X = np.array(X)
    plt.imshow(X)
    plt.title(f'Label: {y}')
    plt.show()

    # Patchify
    patch_size = 7

    patch_mnist = partial(patchify, P=patch_size)
    mnist_transforms = T.Compose([
    T.ToTensor(),
    patch_mnist
])
    transforms = T.Compose([
        T.ToTensor(),
        patch_mnist
    ])

    dataset = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True, 
                                         transform=transforms)
    result, y = dataset[0]
    print(result.shape)
    # print(result.shape)
    fig, axs = plt.subplots(4,4)
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.imshow(result[idx].reshape(patch_size, patch_size))
    plt.suptitle(f'Patched input-- Label: {y}')
    plt.show()
