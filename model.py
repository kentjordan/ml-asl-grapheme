from torch import nn
from torch.nn import functional as F

class ASLModel(nn.Module):
    flatten_size = 2880

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=self.conv1.out_channels + 8,
            kernel_size=(3, 3),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=self.conv2.out_channels + 8,
            kernel_size=(3, 3),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=2880, out_features=64)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=32)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=29)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = self.pool3(out)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)
