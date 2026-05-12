import torch

NUM_CLASSES = 38


class Base_Net(torch.nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear -> fully connected
        self.fc1 = torch.nn.Linear(16*64*64, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        # self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv5(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
