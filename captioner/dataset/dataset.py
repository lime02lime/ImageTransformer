import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

MNIST_PATH = '/Users/kenton/data/'

to_tensor = transforms.ToTensor()


if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True)
    X, y = dataset[0]
    print(np.array(X))
    plt.imshow(np.array(X))
    plt.title(f'Label: {y}')
    plt.show()
