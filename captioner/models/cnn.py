import torch
import torch.nn as nn
from torch.nn import functional as F


class CNN_Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # In channels, out channels, kernel size
        in_channels = 1
        N_classes = 10

        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # These numbers are based on trial and error
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, N_classes)

    def forward(self, x):
        # x: B, 1, H, W
        x = self.pool(F.relu(self.conv1(x)))
        # x: B, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    def test_cnn_encoder_forward():
        cnn_encoder = CNN_Encoder()
        x = torch.randn(1, 1, 28, 28)
        assert cnn_encoder(x).shape == (1,10)

    test_cnn_encoder_forward()
