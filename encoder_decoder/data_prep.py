import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os


class GridMNISTDataset(Dataset):
    def __init__(self, mnist_dataset, grid_size=4, cell_size=28, max_digits=8):
        self.mnist = mnist_dataset
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.canvas_size = grid_size * cell_size
        self.max_digits = max_digits

    def __len__(self):
        return len(self.mnist) // (self.max_digits // 4)

    def __getitem__(self, idx):
        canvas = torch.zeros(1, self.canvas_size, self.canvas_size)
        positions = list(range(self.grid_size * self.grid_size))
        random.shuffle(positions)
        num_digits = random.randint(1, self.max_digits)
        selected_positions = positions[:num_digits]
        selected_positions.sort()

        labels = [10]
        sub_images = []

        for pos in selected_positions:
            row = pos // self.grid_size
            col = pos % self.grid_size
            top = row * self.cell_size
            left = col * self.cell_size

            img_idx = random.randint(0, len(self.mnist) - 1)
            digit_img, label = self.mnist[img_idx]
            canvas[:, top:top+self.cell_size, left:left+self.cell_size] = digit_img
            labels.append(label)
            sub_images.append(digit_img)

        labels.append(11)  # STOP token
        return canvas, torch.tensor(labels), sub_images


def create_image_patches(image, patch_size=14):
    patches = []
    img = image.numpy().squeeze()
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    patches = np.array(patches)
    patches = patches.reshape(len(patches), patch_size * patch_size)
    return patches


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    patch_list = [create_image_patches(image) for image in images]
    patches = torch.tensor(np.array(patch_list), dtype=torch.float32)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=12)  # PAD = 12
    return patches, labels


def show_composed_image(dataset, idx):
    img, labels, sub_images = dataset[idx]
    plt.figure(figsize=(4, 4))
    plt.imshow(img.squeeze(), cmap='gray')
    readable_labels = [str(l.item()) if l.item() < 10 else "STOP" for l in labels]
    plt.title("Composed Image\nDigits: " + ", ".join(readable_labels))
    plt.axis("off")
    plt.show()

    n = len(sub_images)
    plt.figure(figsize=(n, 1))
    for i, digit_img in enumerate(sub_images):
        plt.subplot(1, n, i + 1)
        plt.imshow(digit_img.squeeze(), cmap='gray')
        plt.title(str(labels[i].item()))
        plt.axis("off")
    plt.suptitle("Sub-images (in reading order)", fontsize=12)
    plt.tight_layout()
    plt.show()



def save_dataset(dataset, save_path, chunk_size=1000):
    """Save the composed MNIST dataset (canvas images and labels)."""
    dir_name = os.path.dirname(save_path)
    if dir_name:  # Only create directories if a directory is specified
        os.makedirs(dir_name, exist_ok=True)
    
    # Get total size
    total_size = len(dataset)
    
    # Initialize empty tensors
    sample_canvas, sample_labels, _ = dataset[0]
    all_canvases = torch.empty(total_size, *sample_canvas.shape)
    max_label_len = max(len(dataset[i][1]) for i in range(min(100, total_size)))  # Sample first 100 for max length
    all_labels = torch.full((total_size, max_label_len), 12)  # Initialize with PAD token
    
    # Save in chunks to manage memory
    for i in range(0, total_size, chunk_size):
        end_idx = min(i + chunk_size, total_size)
        print(f"Processing items {i} to {end_idx}")
        
        for j in range(i, end_idx):
            canvas, labels, _ = dataset[j]
            all_canvases[j] = canvas
            all_labels[j, :len(labels)] = labels
    
    torch.save({
        'canvases': all_canvases,
        'labels': all_labels
    }, save_path)
    
    print(f"Dataset saved to {save_path}")


def load_dataset_and_loader(load_path, batch_size=64):
    """Load dataset from a .pt file and return a DataLoader using collate_fn."""
    data = torch.load(load_path)
    canvases = data['canvases']
    labels = data['labels']

    class PrecomputedDataset(Dataset):
        def __init__(self, canvases, labels):
            self.canvases = canvases
            self.labels = labels

        def __len__(self):
            return len(self.canvases)

        def __getitem__(self, idx):
            return self.canvases[idx], self.labels[idx], []  # [] = placeholder for sub-images

    dataset = PrecomputedDataset(canvases, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


def create_and_save_datasets(grid_size=4, cell_size=28, max_digits=10):
    # Define the size of the grid (e.g. 4x4) and size of each MNIST cell (28x28)
    grid_size = 4
    cell_size = 28
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    composed_train = GridMNISTDataset(mnist_train, grid_size=grid_size, cell_size=cell_size, max_digits=10)
    composed_test = GridMNISTDataset(mnist_test, grid_size=grid_size, cell_size=cell_size, max_digits=10)
    save_dataset(composed_train, "encoder_decoder/train_file.pt")
    save_dataset(composed_test, "encoder_decoder/test_file.pt")



def main():
    # Define the size of the grid (e.g. 4x4) and size of each MNIST cell (28x28)
    grid_size = 4
    cell_size = 28
    transform = transforms.ToTensor()

    # Load the original MNIST datasets (download if not already present)
    print("Loading MNIST datasets...")
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    print(f"Loaded {len(mnist_train)} training and {len(mnist_test)} test samples.")

    # Compose grid-based datasets from the flat MNIST digits
    print("Creating composed grid datasets...")
    composed_train = GridMNISTDataset(mnist_train, grid_size=grid_size, cell_size=cell_size, max_digits=10)
    composed_test = GridMNISTDataset(mnist_test, grid_size=grid_size, cell_size=cell_size, max_digits=10)
    print(f"Created {len(composed_train)} train and {len(composed_test)} test composed samples.")

    # Save the composed datasets to disk
    print("Saving composed datasets...")
    save_dataset(composed_train, "encoder_decoder/train_file.pt")
    save_dataset(composed_test, "encoder_decoder/test_file.pt")

    # Optional: Create data loaders (if training now)
    # train_loader = DataLoader(composed_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(composed_test, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Visual check: display a composed image and its corresponding digits
    print("Showing an example composed image.")
    show_composed_image(composed_train, 0)



if __name__ == "__main__":
    main()
