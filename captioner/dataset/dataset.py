from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid

from captioner.utils.visualise import visualise_patched_input

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
    assert C == 1, 'Input image must be single-channel'
    X = X.squeeze()
    assert H % P == 0 and W % P == 0, 'Image dimensions must be divisible by '
    'patch size'

    # Reshape and permute to get non-overlapping patches
    patches = X.unfold(0, P, P).unfold(1, P, P)  # shape: (H/P, W/P, P, P)
    patches = patches.contiguous().view(-1, P*P)  # Flatten each patch

    return patches

# MNIST Classification --------------------------------------------------------


def make_mnist_dataset(patch=False, patch_size=None):

    if patch:
        assert patch_size is not None
        patch_mnist = partial(patchify, P=patch_size)
        mnist_transforms = T.Compose([
            T.ToTensor(),
            patch_mnist,
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

# MNIST Grid Captioning -------------------------------------------------------


class MNISTCaptioningDataset:
    def __init__(self, mnist_path, train: bool = True, transform=None, return_empty_labels=True):
        self.base_dataset = torchvision.datasets.MNIST(
            mnist_path, train=train, download=True,
        )
        self.bos_token_id = 11
        self.eos_token_id = 11
        self.empty_token_id = 10
        self.pad_token_id = 10

        self.im_size = 224
        self.prob_number = 0.4
        self.transform = transform
        self.zero_image = torch.zeros((1, 28, 28))


        self.return_empty_labels = return_empty_labels
        self.grid_size = 8

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Max length L_max self.grid_size * self.grid_size

        numbers: 8x8 grid of MNIST digits, shape (1, self.im_size, self.im_size)
            value range is [0, 1]
        labels: torch.tensor of variable length, shape (1- L_max)
        """
        torch.manual_seed(idx)

        num_bool = torch.rand(self.grid_size * self.grid_size) > (1 - self.prob_number)
        numbers = []
        labels = []

        for i in num_bool:
            if i:
                X, y = self.base_dataset[
                    torch.randint(
                        0, len(self.base_dataset), (1,),
                    ).item()
                ]
                numbers.append(to_tensor(X))
                labels.append(y)
            else:
                numbers.append(self.zero_image)
                if self.return_empty_labels:
                    labels.append(self.empty_token_id)

        # End sequence token
        # labels.append(self.end_token)
        numbers = torch.stack(numbers)

        # Somehow becomes 3 channeled, set to single channeled
        # Shape is (1, 224, 224)
        numbers = make_grid(numbers, nrow=8, padding=0)[0:1]
        y = torch.tensor(labels)
        if self.transform is not None:
            # Output shape is (N = (224/P)**2, P*P)
            numbers = self.transform(numbers)
        return numbers, y
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader
        """
        # List of tuples (image, label)
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)
        return images, labels


def make_mnist_captioning_dataset(mnist_path, patch=False, patch_size=None, return_empty_labels = True):

    if patch:
        assert patch_size is not None
        # TODO scale between -0.5, 0.5
        patch_mnist = partial(patchify, P=patch_size)
        mnist_transforms = patch_mnist 
    else:
        mnist_transforms = None

    train_ds = MNISTCaptioningDataset(
        mnist_path, 
        train=True,  
        transform=mnist_transforms, 
        return_empty_labels=return_empty_labels
    )

    val_ds = MNISTCaptioningDataset(
        mnist_path, 
        train=False,  
        transform=mnist_transforms, 
        return_empty_labels = return_empty_labels
    )

    return train_ds, val_ds


def label_tensor_to_string(labels: torch.Tensor) -> str:
    labels = labels.tolist()
    str_list = []
    for label in labels:
        if label == 11:
            str_list.append('<start>')
        elif label == 12:
            str_list.append('<end>')
        elif label == 10:
            pass
        else:
            str_list.append(str(label))
    return ','.join(str_list)


if __name__ == '__main__':
    def show_image():
        dataset = torchvision.datasets.MNIST(
            MNIST_PATH, train=True, download=True,
        )
        X, y = dataset[0]
        X = np.array(X)
        plt.imshow(X)
        plt.title(f'Label: {y}')
        plt.show()
        return

    def test_patch():
        # Patchify
        patch_size = 7

        patch_mnist = partial(patchify, P=patch_size)
        transforms = T.Compose([
            T.ToTensor(),
            patch_mnist,
        ])

        dataset = torchvision.datasets.MNIST(
            MNIST_PATH, train=True, download=True,
            transform=transforms,
        )
        result, y = dataset[0]
        print(result.shape)
        # print(result.shape)
        fig, axs = plt.subplots(4, 4)
        axs = axs.flatten()
        for idx, ax in enumerate(axs):
            ax.imshow(result[idx].reshape(patch_size, patch_size))
        plt.suptitle(f'Patched input-- Label: {y}')
        plt.show()
        return

    def test_patched_captioning():

        dataset = MNISTCaptioningDataset(
            MNIST_PATH, train=True, transform=None,
        )
        plt.imshow(dataset[0][0].squeeze())
        plt.show()

        patch_size = 16
        patch_mnist = partial(patchify, P=patch_size)
        dataset = MNISTCaptioningDataset(
            MNIST_PATH, train=True, transform=patch_mnist, return_empty_labels = False
        )
        result, y = dataset[0]
        print(result.shape)
        visualise_patched_input(result, label_tensor_to_string(y), patch_size)

    test_patched_captioning()
