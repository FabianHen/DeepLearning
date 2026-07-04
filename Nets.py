import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

NUM_CLASSES = 38


class Base_Net(nn.Module):
    """Baseline convolutional classifier with a small fully connected head."""

    def __init__(self):
        """Initialize the baseline convolutional network."""
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
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """Run a forward pass through the baseline network."""
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
    """Pure convolutional feature extractor followed by a dense classifier."""

    def __init__(self):
        """Initialize the deeper convolutional network."""
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
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """Run a forward pass through the convolutional-only model."""
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
    """Convolutional stem with a deeper fully connected classifier head."""

    def __init__(self):
        """Initialize the dense-head network."""
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
        self.fc7 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """Run a forward pass through the dense classifier."""
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
    """Network variant that applies pooling after nearly every convolution."""

    def __init__(self):
        """Initialize the pooling-heavy network."""
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
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """Run a forward pass through the pooling-heavy architecture."""
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
    """Convolutional classifier with dropout regularization in the head."""

    def __init__(self):
        """Initialize the dropout-regularized network."""
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
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """Run a forward pass with dropout applied in the dense head."""
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
    """Convolutional classifier that uses batch normalization throughout."""

    def __init__(self):
        """Initialize the batch-normalized network."""
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

    def forward(self, x):
        """Run a forward pass with batch normalization in each block."""
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
    """Batch-normalized network with pooled convolutional blocks and dense head."""

    def __init__(self):
        """Initialize the pooled and dense batch-normalized network."""
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

    def forward(self, x):
        """Run a forward pass through the pooled BN architecture."""
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
    """Batch-normalized variant that keeps one extra convolutional block."""

    def __init__(self):
        """Initialize the extended batch-normalized network."""
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

    def forward(self, x):
        """Run a forward pass through the extended BN convolutional model."""
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
    """Batch-normalized network with pooling, an extra conv block, and dropout."""

    def __init__(self):
        """Initialize the batch-normalized dropout network."""
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

    def forward(self, x):
        """Run a forward pass with batch normalization and dropout."""
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
    """Wider batch-normalized dropout network with a larger convolutional stem."""

    def __init__(self):
        """Initialize the wider V2 batch-normalized dropout network."""
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

    def forward(self, x):
        """Run a forward pass through the wider V2 architecture."""
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
    """Transfer learning model based on a pretrained ResNet-50 backbone."""

    def __init__(self):
        """Initialize the pretrained ResNet-50 classifier head."""
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze the full backbone first, then unfreeze only the last block.
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x):
        """Run a forward pass through the fine-tuned ResNet-50 model."""
        return self.model(x)

class Student_Net(nn.Module):
    """Student model for knowledge distillation."""

    def __init__(self):
        """Initialize the student model."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((32, 32))
        # Linear -> fully connected
        self.fc1 = nn.Linear(16*32*32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """Run a forward pass through the baseline network."""
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.global_avg_pool(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x