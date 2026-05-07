import csv
import shutil
from pathlib import Path


DATASETS = [
    ("images/EuroSAT", "images/EuroSAT_test"),
    ("images/EuroSATallBands", "images/EuroSATallBands_test"),
]


def read_filenames(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if "Filename" not in reader.fieldnames:
            raise ValueError(f"'Filename' column is missing in {csv_path}")
        return [row["Filename"].strip() for row in reader if row.get("Filename", "").strip()]


def move_test_images(dataset_root: Path, target_root: Path) -> tuple[int, int, list[str]]:
    test_csv = dataset_root / "test.csv"
    filenames = read_filenames(test_csv)

    moved = 0
    skipped = 0
    missing: list[str] = []

    for relative_name in filenames:
        source_path = dataset_root / relative_name
        destination_path = target_root / relative_name
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.exists():
            shutil.move(str(source_path), str(destination_path))
            moved += 1
        elif destination_path.exists():
            skipped += 1
        else:
            missing.append(relative_name)

    return moved, skipped, missing


def main() -> None:
    project_root = Path(__file__).resolve().parent

    for dataset_folder, target_folder in DATASETS:
        dataset_root = project_root / dataset_folder
        target_root = project_root / target_folder
        target_root.mkdir(parents=True, exist_ok=True)

        moved, skipped, missing = move_test_images(dataset_root, target_root)

        print(
            f"{dataset_root.name}: moved={moved}, skipped={skipped}, missing={len(missing)}")
        if missing:
            for relative_name in missing:
                print(f"  missing: {relative_name}")


if __name__ == "__main__":
    main()
