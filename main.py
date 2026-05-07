import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torchinfo
import matplotlib.pyplot as plt

batch_size = 64
NUM_WORKERS = 4
DATA_ROOT = Path("images/EuroSAT")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# define a transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor(),  # convert to tensor & scale to [0,1]
     # for each channel
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


class EuroSATCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: Path, image_root: Path, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []

        with csv_path.open(newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                filename = row.get("Filename", "").strip()
                label = row.get("Label", "").strip()
                if not filename or not label:
                    continue
                self.samples.append((filename, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, label = self.samples[index]
        image_path = self.image_root / relative_path
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


trainset = EuroSATCSVDataset(
    DATA_ROOT / "train.csv", DATA_ROOT, transform=transform)
valset = EuroSATCSVDataset(
    DATA_ROOT / "validation.csv", DATA_ROOT, transform=transform)
testset = EuroSATCSVDataset(DATA_ROOT / "test.csv",
                            DATA_ROOT, transform=transform)

classes = ('AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake')

NUM_CLASSES = 10


class Net(torch.nn.Module):
    # define layer
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Linear -> fully connected
        self.fc1 = torch.nn.Linear(16*16*16, 128)
        # self.fc2 = torch.nn.Linear(120, 84)
        self.fc2 = torch.nn.Linear(128, NUM_CLASSES)  # 10 classes

    # define data flow / architecture
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  # output=32 * 32*32
        x = self.pool(x)                            # output=32 * 16*16
        x = torch.nn.functional.relu(self.conv2(x))  # output=16 * 16*16
        x = self.pool(x)                            # output=16 *  8* 8
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc2(x)
        return x


def main():
    print("cuda available: ", torch.cuda.is_available())
    print("#gpus: ", torch.cuda.device_count())
    print("#threads: ", torch.get_num_threads())
    print(device)

    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    network = Net().to(device)
    print(torchinfo.summary(network, input_size=(1, 3, 64, 64), device=device))
    network = network.to(device)

    num_epochs = 4
    optimizer = torch.optim.Adam(network.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        network.train()
        running_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)

        network.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = network(inputs)
                val_loss = loss_function(outputs, labels)
                val_running_loss += val_loss.item()

                predictions = torch.argmax(outputs, dim=1)
                total_predictions += labels.size(0)
                correct_predictions += (predictions == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_dataloader)
        val_accuracy = 100.0 * correct_predictions / total_predictions

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )

    print('Finished Training')
    network.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # inputs, labels = data
            # calculate outputs by running images through the network
            outputs = network(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the test images: {100 * correct // total} %')


if __name__ == "__main__":
    main()
