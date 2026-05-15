from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torchinfo
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix

from Nets import Base_Net, Dense_Net, Conv_Net, Pooling_Net
from Dataset import ImageFolderDataset

NUM_EPOCHS = 15
BATCH_SIZE = 64
NUM_WORKERS = 4
DATA_ROOT = Path("images/PatternNet_Images")
NET_CLASS = Base_Net
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

CLASSES = (
    'airplane',
    'baseball_field',
    'basketball_court',
    'beach',
    'bridge',
    'cemetery',
    'chaparral',
    'christmas_tree_farm',
    'closed_road',
    'coastal_mansion',
    'crosswalk',
    'dense_residential',
    'ferry_terminal',
    'football_field',
    'forest',
    'freeway',
    'golf_course',
    'harbor',
    'intersection',
    'mobile_home_park',
    'nursing_home',
    'oil_gas_field',
    'oil_well',
    'overpass',
    'parking_lot',
    'parking_space',
    'railway',
    'river',
    'runway',
    'runway_marking',
    'shipping_yard',
    'solar_panel',
    'sparse_residential',
    'storage_tank',
    'swimming_pool',
    'tennis_court',
    'transformer_station',
    'wastewater_treatment_plant',
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def main():
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders()

    network = NET_CLASS().to(device)
    torchinfo.summary(network, input_size=(1, 3, 256, 256), device=device)

    train_and_validate(train_dataloader, val_dataloader, network)
    test(test_dataloader, network)


def _dataset_paths(dataset):
    return {str(path.resolve()) for path, _ in dataset.samples}


def _label_counts(dataset):
    counts = np.zeros(len(CLASSES), dtype=int)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def validate_split_integrity(trainset, valset, testset):
    train_paths = _dataset_paths(trainset)
    val_paths = _dataset_paths(valset)
    test_paths = _dataset_paths(testset)

    overlap_train_val = train_paths & val_paths
    overlap_train_test = train_paths & test_paths
    overlap_val_test = val_paths & test_paths

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            "Data leakage detected: train/val/test contain overlapping file paths."
        )

    print("Split integrity check passed: no overlapping file paths across train/val/test.")

    for split_name, dataset in (("train", trainset), ("val", valset), ("test", testset)):
        counts = _label_counts(dataset)
        print(
            f"{split_name} split size: {len(dataset)} | "
            f"min/class: {counts.min()} | max/class: {counts.max()}"
        )


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

    # Create base dataset and split it deterministically per class
    full_dataset = ImageFolderDataset(DATA_ROOT)
    trainset, valset, testset = full_dataset.stratified_split(
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        seed=SPLIT_SEED,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
    )

    validate_split_integrity(trainset, valset, testset)

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
    writer = SummaryWriter(f"runs/{NET_CLASS.__name__}")

    optimizer = torch.optim.Adam(network.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    log_sample_images(writer, train_dataloader)

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
        val_epoch_loss, val_accuracy, all_targets, all_preds = validate(
            val_dataloader, network, loss_function)

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/val", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        log_confusion_matrix(writer, all_targets, all_preds, epoch)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )
    writer.close()
    print('Finished Training')


def log_sample_images(writer, train_dataloader):
    images, labels = next(iter(train_dataloader))
    images = images[:16].clone()
    labels = labels[:16]

    images = images * 0.5 + 0.5
    images = images.clamp(0.0, 1.0)

    grid = make_grid(images, nrow=4)
    writer.add_image("Samples/train_images", grid)
    writer.add_text(
        "Samples/train_labels",
        ", ".join(CLASSES[label.item()] for label in labels),
    )
    try:
        writer.flush()
    except Exception:
        pass


# Validation function to evaluate the model on the validation set
def validate(val_dataloader, network, loss_function):
    network.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_targets = []
    all_preds = []

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

            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(predictions.cpu().tolist())

    val_epoch_loss = val_running_loss / len(val_dataloader)
    val_accuracy = 100.0 * correct_predictions / total_predictions
    return val_epoch_loss, val_accuracy, all_targets, all_preds


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(32, 32))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion matrix')
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    return fig


def log_confusion_matrix(writer, targets, preds, epoch):
    cm = confusion_matrix(targets, preds)
    fig = plot_confusion_matrix(cm, CLASSES)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    import PIL.Image as PILImage
    im = PILImage.open(buf).convert('RGB')
    im = np.array(im).transpose(2, 0, 1)
    writer.add_image('ConfusionMatrix', im, epoch)


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

    test_accuracy = 100.0 * correct / total
    print(f"Accuracy of the network on the test images: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
