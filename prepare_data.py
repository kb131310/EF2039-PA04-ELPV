from pathlib import Path
from zipfile import ZipFile

import pandas as pd

folder = Path(__file__).resolve().parent
zip_path = folder / "elpv-dataset-master.zip"
labels_path = folder / "elpv-dataset-master/src/elpv_dataset/data/labels.csv"

if not labels_path.is_file():
    with ZipFile(zip_path) as archive:
        archive.extractall(folder)

labels = pd.read_csv(
    labels_path,
    sep=r"\s+",
    header=None,
    names=["image_path", "original_label", "cell_type"],
)

# Keep only the normal (0) and defective (1) labels.
binary = labels.loc[labels["original_label"].isin([0.0, 1.0])].copy()
binary["binary_label"] = binary["original_label"].astype(int)

# Set aside the same validation and test images for every experiment.
parts = []
for _, group in binary.groupby("binary_label"):
    shuffled = group.sample(frac=1, random_state=42)
    parts.extend([
        shuffled.iloc[:100].assign(split="test"),
        shuffled.iloc[100:150].assign(split="validation"),
        shuffled.iloc[150:].assign(split="train"),
    ])

splits = pd.concat(parts, ignore_index=True)
splits.to_csv(folder / "splits.csv", index=False)

print(pd.crosstab(splits["split"], splits["binary_label"]))
print("Saved splits.csv")