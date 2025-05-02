import matplotlib.pyplot as plt
import torch
import torchvision

from captioner.dataset import MNIST_PATH, to_tensor
from captioner.models import CNN_Encoder
from captioner.utils import get_device

# query_encoder.load_state_dict(checkpoint['query_encoder_state_dict'])
val_ds = torchvision.datasets.MNIST(
    MNIST_PATH,
    train=False,
    download=True,
    transform=to_tensor,
)
encoder = CNN_Encoder()
device = get_device()
checkpoint = torch.load(
    'checkpoints/20250428_145147.pth', map_location=device, weights_only=True,
)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
X, y = val_ds[0]

with torch.no_grad():
    scores = encoder(val_ds[0][0].reshape(1, 1, 28, 28))
    print(scores, y)
    pred = scores.argmax()
    plt.title(f'True label: {y}, Predicted label: {pred}')
    plt.imshow(X[0], cmap='grey')
    plt.show()
