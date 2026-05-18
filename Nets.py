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
    
class Dropout_Net(torch.nn.Module):
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
        
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate
        
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
        x = self.dropout(x)  # Apply dropout
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class BatchNorm_Net(torch.nn.Module):
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
        
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.bn5 = torch.nn.BatchNorm2d(16)

        # Linear -> fully connected
        self.fc1 = torch.nn.Linear(16*64*64, 128)
        self.bn_fc1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class Best_Net(torch.nn.Module):
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
        
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.bn5 = torch.nn.BatchNorm2d(16)

        # Linear -> fully connected
        self.fc1 = torch.nn.Linear(16*32*32, 128)
        self.bn_fc1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.bn_fc3 = torch.nn.BatchNorm1d(128)
        self.fc4 = torch.nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn_fc2(self.fc2(x))) 
        x = torch.nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x

class BestConvDropout_Net(torch.nn.Module):
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
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.bn6 = torch.nn.BatchNorm2d(16)

        # Linear -> fully connected
        self.fc1 = torch.nn.Linear(16*32*32, 128)
        self.bn_fc1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.bn_fc3 = torch.nn.BatchNorm1d(128)
        self.fc4 = torch.nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn6(self.conv6(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn_fc2(self.fc2(x))) 
        x = torch.nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class BestConv_Net(torch.nn.Module):
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
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.bn6 = torch.nn.BatchNorm2d(16)

        self.dropout = torch.nn.Dropout(p=0.5)

        # Linear -> fully connected
        self.fc1 = torch.nn.Linear(16*32*32, 128)
        self.bn_fc1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.bn_fc3 = torch.nn.BatchNorm1d(128)
        self.fc4 = torch.nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn6(self.conv6(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class BestFeatureMaps_Net(torch.nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.bn6 = torch.nn.BatchNorm2d(32)

        self.dropout = torch.nn.Dropout(p=0.5)

        # Linear -> fully connected
        self.fc1 = torch.nn.Linear(32*32*32, 128)
        self.bn_fc1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn_fc2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.bn_fc3 = torch.nn.BatchNorm1d(128)
        self.fc4 = torch.nn.Linear(128, NUM_CLASSES)

    # define data flow / architecture
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.bn6(self.conv6(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)
        return x