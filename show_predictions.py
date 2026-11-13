import argparse
import csv
from pathlib import Path
from random import Random


LABELS = {"0": "Normal", "1": "Defect"}
OUTCOMES = {
    ("1", "1"): "Correct defect",
    ("1", "0"): "Missed defect",
    ("0", "1"): "False alarm",
    ("0", "0"): "Correct normal",
}
IMAGES_PER_PAGE = 8


def select_examples(rows, seed):
    groups = {outcome: [] for outcome in OUTCOMES}
    for row in rows:
        outcome = (row["true_label"], row["predicted_label"])
        groups[outcome].append(row)

    rng = Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    examples = [row for group in groups.values() for row in group[:2]]
    remaining = [row for group in groups.values() for row in group[2:]]
    examples.extend(rng.sample(remaining, IMAGES_PER_PAGE - len(examples)))
    return examples


def save_gallery(examples, data_dir, title, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8.6, 5.9), constrained_layout=True)
    grid = figure.add_gridspec(
        4, 4, height_ratios=(1, 0.29, 1, 0.29), hspace=0.06
    )
    figure.suptitle(title, fontsize=13, weight="bold")

    for index, row in enumerate(examples):
        grid_row, column = divmod(index, 4)
        image_axis = figure.add_subplot(grid[2 * grid_row, column])
        label_axis = figure.add_subplot(grid[2 * grid_row + 1, column])

        actual = row["true_label"]
        predicted = row["predicted_label"]
        outcome = OUTCOMES[(actual, predicted)]
        score = float(row["defect_score"])
        image_path = data_dir / row["image_path"]

        image_axis.imshow(plt.imread(image_path), cmap="gray", vmin=0, vmax=1)
        image_axis.set_axis_off()
        image_axis.set_title(image_path.name, fontsize=8, pad=4)

        label_axis.set_axis_off()
        label_axis.text(
            0.5,
            0.5,
            f"Actual: {LABELS[actual]}  |  Pred: {LABELS[predicted]}\n"
            f"Defect score: {score:.2%}\n{outcome}",
            transform=label_axis.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.3,
            color="#24313b" if actual == predicted else "#8c3214",
        )

    figure.savefig(destination, dpi=300, facecolor="white")
    plt.close(figure)


def save_pages(rows, data_dir, title, destination):
    ordered = sorted(rows, key=lambda row: row["image_path"])
    page_count = (len(ordered) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    digits = max(2, len(str(page_count)))
    destination.mkdir(exist_ok=True)

    for page, start in enumerate(range(0, len(ordered), IMAGES_PER_PAGE), start=1):
        page_rows = ordered[start:start + IMAGES_PER_PAGE]
        page_path = destination / f"page_{page:0{digits}d}.png"
        page_title = f"{title} · Page {page}/{page_count}"
        save_gallery(page_rows, data_dir, page_title, page_path)
        print(f"Saved {page_path.name} ({page}/{page_count})", flush=True)

    print(f"Saved all {len(ordered)} test images to {destination}")


def main():
    parser = argparse.ArgumentParser(
        description="Show ELPV test predictions, eight images per page."
    )
    parser.add_argument("--condition", type=str.upper, default="C")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", dest="all_images", action="store_true")
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent
    data_dir = folder / "elpv-dataset-master/src/elpv_dataset/data"
    result_dir = folder / "results" / f"seed_{args.seed}"
    predictions_path = result_dir / "predictions.csv"

    if not predictions_path.is_file():
        parser.error(f"Predictions file not found: {predictions_path}")

    with predictions_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["condition"] == args.condition
        ]

    if not rows:
        parser.error(f"No predictions found for condition {args.condition}.")

    title = f"ELPV test predictions · Condition {args.condition}"
    if args.all_images:
        destination = result_dir / f"prediction_pages_{args.condition}"
        save_pages(rows, data_dir, title, destination)
        return

    if len(rows) < IMAGES_PER_PAGE:
        parser.error("At least eight predictions are needed for the example page.")

    examples = select_examples(rows, args.seed)
    destination = result_dir / f"prediction_examples_{args.condition}.png"
    save_gallery(examples, data_dir, title, destination)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()