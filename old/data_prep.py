import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def load_mnist_data():
    """
    Load the MNIST dataset and apply transformations.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images to have values between -1 and 1
    ])

    # Download MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def create_image_patches(image, patch_size=14):
    """
    Given an image, create patches of specified size (patch_size x patch_size).
    """
    patches = []
    img = image.numpy().squeeze()  # Convert to numpy array
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    
    patches = np.array(patches)
    patches = patches.reshape(len(patches), patch_size * patch_size)  # Flatten each patch
    return patches


def collate_fn(batch):
    """
    Collate function to group patches into a batch.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Create patches for each image
    patch_list = []
    for image in images:
        patches = create_image_patches(image)
        patch_list.append(patches)

    # Stack patches for batch
    patches = torch.tensor(np.array(patch_list))
    labels = torch.tensor(np.array(labels))
    return patches, labels


def create_dataloaders(batch_size=256):
    """
    Create DataLoader for training and testing.
    """
    train_dataset, test_dataset = load_mnist_data()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader
