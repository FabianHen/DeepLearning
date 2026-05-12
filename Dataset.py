import csv
from pathlib import Path
from PIL import Image
import torch


DATA_ROOT = Path("images/EuroSAT")


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
