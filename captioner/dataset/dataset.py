import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

MNIST_PATH = '/Users/kenton/data/'

to_tensor = transforms.ToTensor()

def patchify(X: torch.Tensor, P: int) -> torch.Tensor:
    """
    Splits a single-channel image into non-overlapping P×P patches.

    Args:
        X (torch.Tensor): Input image of shape (H, W)
        P (int): Patch side length

    Returns:
        torch.Tensor: Tensor of shape (N_patches, P*P)
    """
    H, W = X.shape
    assert H % P == 0 and W % P == 0, "Image dimensions must be divisible by patch size"

    # Reshape and permute to get non-overlapping patches
    patches = X.unfold(0, P, P).unfold(1, P, P)  # shape: (H/P, W/P, P, P)
    patches = patches.contiguous().view(-1, P*P)  # Flatten each patch

    return patches

if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True)
    X, y = dataset[0]
    X = np.array(X)
    plt.imshow(X)
    plt.title(f'Label: {y}')
    plt.show()

    # Patchify
    patch_size = 7
    result = patchify(torch.tensor(X), patch_size)
    print(result.shape)
    fig, axs = plt.subplots(4,4)
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.imshow(result[idx].reshape(patch_size, patch_size))
    plt.suptitle(f'Patched input-- Label: {y}')
    plt.show()
