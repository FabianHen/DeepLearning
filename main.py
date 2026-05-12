from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torchinfo
from tqdm import tqdm

from Nets import Net
from Dataset import EuroSATCSVDataset

NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 4
DATA_ROOT = Path("images/EuroSAT")

CLASSES = ('AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake')

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def main():
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders()

    network = Net().to(device)
    torchinfo.summary(network, input_size=(1, 3, 64, 64), device=device)

    train_and_validate(train_dataloader, val_dataloader, network)
    test(test_dataloader, network)


# Function to create dataloaders for training, validation, and testing
def get_data_loaders():
    # Define transformations for training and evaluation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Create datasets
    trainset = EuroSATCSVDataset(
        DATA_ROOT / "train.csv", DATA_ROOT, transform=train_transform)
    valset = EuroSATCSVDataset(
        DATA_ROOT / "validation.csv", DATA_ROOT, transform=eval_transform)
    testset = EuroSATCSVDataset(DATA_ROOT / "test.csv",
                                DATA_ROOT, transform=eval_transform)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader, test_dataloader


# Training function to train the model and validate after each epoch
def train_and_validate(train_dataloader, val_dataloader, network):
    optimizer = torch.optim.Adam(network.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        network.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)

        val_epoch_loss, val_accuracy = validate(
            val_dataloader, network, loss_function)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )

    print('Finished Training')


# Validation function to evaluate the model on the validation set
def validate(val_dataloader, network, loss_function):
    network.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc="Validation"):
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
    return val_epoch_loss, val_accuracy


# Test function to evaluate the model on the test set
def test(test_dataloader, network):
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Testing"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the test images: {100 * correct // total} %')


if __name__ == "__main__":
    main()
