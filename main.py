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

from Nets import Base_Net, Dense_Net, Conv_Net, Pooling_Net, Dropout_Net, BatchNorm_Net, BatchNorm_Dense_Pool_Net, BatchNorm_Dense_Pool_Conv_Net, BatchNorm_Dense_Pool_Conv_Dropout_Net, BatchNorm_Dense_Pool_Conv_Dropout_V2_Net, TransferNet, Student_Net
from Dataset import ImageFolderDataset

NUM_EPOCHS = 15
BATCH_SIZE = 64
NUM_WORKERS = 4
DATA_ROOT = Path("images/PatternNet_Images")
CHECKPOINT_DIR = Path("checkpoints")
NET_CLASS = Student_Net
TEACHER_CLASS = TransferNet
TEACHER_CHECKPOINT = CHECKPOINT_DIR / f"{TEACHER_CLASS.__name__}.pth"
DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA = 0.2
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
    """Run the full training, validation, and test pipeline."""
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders()

    network = NET_CLASS().to(device)
    torchinfo.summary(network, input_size=(1, 3, 256, 256), device=device)

    writer = SummaryWriter(f"runs/{NET_CLASS.__name__}")
    train_and_validate(train_dataloader, val_dataloader, network, writer)
    # save_model(network)
    test_accuracy = test(test_dataloader, network)
    writer.add_scalar("Accuracy/test", test_accuracy)
    writer.close()


def distill_main():
    """Train the student network via knowledge distillation from the pretrained teacher."""
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders()

    teacher = load_teacher()
    student = NET_CLASS().to(device)
    torchinfo.summary(student, input_size=(1, 3, 256, 256), device=device)

    writer = SummaryWriter(f"runs/{NET_CLASS.__name__}_distilled")
    train_and_validate_distillation(train_dataloader, val_dataloader, teacher, student, writer)
    test_accuracy = test(test_dataloader, student)
    writer.add_scalar("Accuracy/test", test_accuracy)
    writer.close()


def _dataset_paths(dataset):
    """Return the resolved file paths contained in a dataset."""
    return {str(path.resolve()) for path, _ in dataset.samples}


def _label_counts(dataset):
    """Count the number of samples per class in a dataset."""
    counts = np.zeros(len(CLASSES), dtype=int)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def validate_split_integrity(trainset, valset, testset):
    """Check that the train/validation/test splits do not overlap."""
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


def get_data_loaders():
    """Build transformed train, validation, and test dataloaders."""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Keep evaluation deterministic and free from augmentation.
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Build the base dataset once, then split it deterministically per class.
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

    # Enable shuffling only for training; evaluation keeps a fixed order.
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


def train_and_validate(train_dataloader, val_dataloader, network, writer):
    """Train the network and validate it after every epoch."""

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
        # Validation returns both scalar metrics and full predictions for the
        # confusion matrix.
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
    print('Finished Training')


def save_model(network, name=None):
    """Save the trained network's weights to the checkpoint directory."""
    name = name or NET_CLASS.__name__
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{name}.pth"
    torch.save(network.state_dict(), checkpoint_path)
    print(f"Saved model weights to {checkpoint_path}")


def load_teacher():
    """Load the pretrained teacher network and freeze it for inference only."""
    teacher = TEACHER_CLASS().to(device)
    teacher.load_state_dict(torch.load(TEACHER_CHECKPOINT, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """Combine hard-label cross-entropy with soft-target KL-divergence distillation loss."""
    student_loss = torch.nn.functional.cross_entropy(student_logits, labels)
    soft_distillation_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / temperature, dim=1),
        torch.nn.functional.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
    # Scale by T^2 so the soft-target gradient magnitude doesn't shrink as
    # temperature increases (Hinton et al., 2015).
    return alpha * student_loss + (1 - alpha) * (temperature ** 2) * soft_distillation_loss


def train_and_validate_distillation(train_dataloader, val_dataloader, teacher, student, writer,
                                     temperature=DISTILL_TEMPERATURE, alpha=DISTILL_ALPHA):
    """Train the student network via knowledge distillation from the frozen teacher."""

    optimizer = torch.optim.Adam(student.parameters())
    val_loss_function = torch.nn.CrossEntropyLoss()

    log_sample_images(writer, train_dataloader)

    for epoch in range(NUM_EPOCHS):
        student.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Distillation Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            optimizer.zero_grad()
            student_logits = student(inputs)
            loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        val_epoch_loss, val_accuracy, all_targets, all_preds = validate(
            val_dataloader, student, val_loss_function)

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/val", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        log_confusion_matrix(writer, all_targets, all_preds, epoch)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )
    print('Finished Distillation Training')


def log_sample_images(writer, train_dataloader):
    """Log a small grid of training samples and their labels to TensorBoard."""
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


def validate(val_dataloader, network, loss_function):
    """Evaluate the model on the validation split."""
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
    """Render a confusion matrix figure for TensorBoard logging."""
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
    """Convert the current validation predictions into a TensorBoard image."""
    cm = confusion_matrix(targets, preds)
    fig = plot_confusion_matrix(cm, CLASSES)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    import PIL.Image as PILImage
    im = PILImage.open(buf).convert('RGB')
    im = np.array(im).transpose(2, 0, 1)
    writer.add_image('ConfusionMatrix', im, epoch)


def test(test_dataloader, network):
    """Evaluate the final model on the held-out test split."""
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
    return test_accuracy


if __name__ == "__main__":
    main()  # pretrain the teacher (TransferNet) and save its checkpoint
    # distill_main()  # train the student via knowledge distillation
