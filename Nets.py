import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

NUM_CLASSES = 38


class Base_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((32, 32))
        # Linear -> fully connected
        self.fc1 = nn.Linear(16*32*32, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv5(x))
        x = self.global_avg_pool(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear -> fully connected
        self.fc1 = nn.Linear(16*64*64, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv7(x))
        x = nn.functional.relu(self.conv8(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Dense_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear -> fully connected
        self.fc1 = nn.Linear(16*64*64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv5(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        x = nn.functional.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class Pooling_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear -> fully connected
        self.fc1 = nn.Linear(16*16*16, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv5(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Dropout_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer with 50% dropout rate
        self.dropout = nn.Dropout(p=0.5)

        # Linear -> fully connected
        self.fc1 = nn.Linear(16*64*64, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv5(x))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BatchNorm_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)

        # Linear -> fully connected
        self.fc1 = nn.Linear(16*64*64, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x


class BatchNorm_Dense_Pool_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)

        # Linear -> fully connected
        self.fc1 = nn.Linear(16*32*32, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x


class BatchNorm_Dense_Pool_Conv_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(16)

        # Linear -> fully connected
        self.fc1 = nn.Linear(16*32*32, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn6(self.conv6(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x


class BatchNorm_Dense_Pool_Conv_Dropout_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(16)

        self.dropout = nn.Dropout(p=0.5)

        # Linear -> fully connected
        self.fc1 = nn.Linear(16*32*32, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn6(self.conv6(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x


class BatchNorm_Dense_Pool_Conv_Dropout_V2_Net(nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(p=0.5)

        # Linear -> fully connected
        self.fc1 = nn.Linear(32*32*32, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = nn.functional.relu(self.bn6(self.conv6(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x


class TransferNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)
