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
        self.image_cache = {}  # Cache for images in RAM

        with csv_path.open(newline="", encoding="utf-8") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                filename = row.get("Filename", "").strip()
                label = row.get("Label", "").strip()
                if not filename or not label:
                    continue
                self.samples.append((filename, int(label)))

        # Preload all images into RAM cache
        print(f"Preloading {len(self.samples)} images into cache...")
        for idx, (relative_path, label) in enumerate(self.samples):
            image_path = self.image_root / relative_path
            image = Image.open(image_path).convert("RGB")
            self.image_cache[idx] = image
        print("Image cache ready!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Get image from cache instead of loading from disk
        # Copy to avoid modifying cached image
        image = self.image_cache[index].copy()
        _, label = self.samples[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
