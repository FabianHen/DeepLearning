import csv
import random
from pathlib import Path
from PIL import Image
import torch


IMAGE_EXTENSIONS = {".jpg"}
SAMPLES_PER_CLASS = 100
SAMPLE_SELECTION_SEED = 42


class ImageFolderDataset(torch.utils.data.Dataset):
    """Dataset wrapper that loads class-organized images from a directory tree.

    The dataset scans one subdirectory per class, stores `(image_path, label)`
    pairs, and optionally applies a transform when samples are fetched.
    """

    def __init__(self, image_root: Path, transform=None, samples=None, class_to_idx=None):
        """Create a dataset from an image root folder.

        Args:
            image_root: Root directory that contains one folder per class.
            transform: Optional torchvision transform applied to every image.
            samples: Optional precomputed `(path, label)` list for subsets.
            class_to_idx: Optional mapping that fixes the class ordering.
        """
        self.image_root = Path(image_root)
        self.transform = transform

        if class_to_idx is None:
            class_names = [
                entry.name for entry in sorted(self.image_root.iterdir())
                if entry.is_dir()
            ]
            self.class_to_idx = {class_name: index for index,
                                 class_name in enumerate(class_names)}
        else:
            self.class_to_idx = dict(class_to_idx)

        self.samples = samples if samples is not None else self._scan_samples()

    def _scan_samples(self):
        """Collect image paths for every class folder.

        Returns:
            A list of `(image_path, class_index)` tuples.
        """
        samples = []
        for class_name, class_index in self.class_to_idx.items():
            class_dir = self.image_root / class_name
            if not class_dir.exists():
                continue

            class_samples = []
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    class_samples.append((image_path, class_index))

            # Limit samples per class if specified
            if SAMPLES_PER_CLASS is not None and len(class_samples) > SAMPLES_PER_CLASS:
                # Use a reproducible random subset to avoid lexicographic sampling bias.
                class_rng = random.Random(SAMPLE_SELECTION_SEED + class_index)
                class_samples = class_rng.sample(
                    class_samples, SAMPLES_PER_CLASS)
                class_samples.sort(key=lambda item: str(item[0]))

            samples.extend(class_samples)

        if not samples:
            raise ValueError(f"No images found in {self.image_root}")

        return samples

    def subset(self, indices, transform=None):
        """Create a dataset subset from the given sample indices.

        Args:
            indices: Indices into the current dataset samples.
            transform: Optional override transform for the returned subset.

        Returns:
            A new `ImageFolderDataset` backed by the selected samples.
        """
        subset_samples = [self.samples[index] for index in indices]
        return ImageFolderDataset(
            self.image_root,
            transform=self.transform if transform is None else transform,
            samples=subset_samples,
            class_to_idx=self.class_to_idx,
        )

    def stratified_split(
        self,
        train_ratio,
        val_ratio,
        test_ratio,
        seed=42,
        train_transform=None,
        val_transform=None,
        test_transform=None,
    ):
        """Split the dataset into train, validation, and test subsets.

        The split is performed per class so each subset preserves the class
        distribution as closely as possible.
        """
        if not torch.isclose(
            torch.tensor(train_ratio + val_ratio +
                         test_ratio, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        ):
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0")

        rng = random.Random(seed)
        train_samples = []
        val_samples = []
        test_samples = []

        samples_by_class = {}
        for sample in self.samples:
            image_path, label = sample
            samples_by_class.setdefault(label, []).append(sample)

        for class_label, class_samples in samples_by_class.items():
            rng.shuffle(class_samples)
            total = len(class_samples)
            train_count = int(total * train_ratio)
            val_count = int(total * val_ratio)
            test_count = total - train_count - val_count

            train_samples.extend(class_samples[:train_count])
            val_samples.extend(
                class_samples[train_count:train_count + val_count])
            test_samples.extend(
                class_samples[train_count + val_count:train_count + val_count + test_count])

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        rng.shuffle(test_samples)

        return (
            ImageFolderDataset(self.image_root, transform=train_transform if train_transform is not None else self.transform,
                               samples=train_samples, class_to_idx=self.class_to_idx),
            ImageFolderDataset(self.image_root, transform=val_transform if val_transform is not None else self.transform,
                               samples=val_samples, class_to_idx=self.class_to_idx),
            ImageFolderDataset(self.image_root, transform=test_transform if test_transform is not None else self.transform,
                               samples=test_samples, class_to_idx=self.class_to_idx),
        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load and return a single image-label pair."""
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
