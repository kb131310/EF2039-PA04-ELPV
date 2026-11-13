import argparse
import csv
from pathlib import Path
from random import Random

import tensorflow as tf


CONDITIONS = {
    "A": (20, 20),
    "B": (50, 50),
    "C": (100, 100),
    "E": (150, 50),
    "F": (180, 20),
}
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
THRESHOLD = 0.5


def load_image(path, label):
    image = tf.io.decode_png(tf.io.read_file(path), channels=1)
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, label


def image_batches(rows, data_dir, shuffle_seed=None):
    paths = [str(data_dir / row["image_path"]) for row in rows]
    labels = [float(row["binary_label"]) for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle_seed is not None:
        dataset = dataset.shuffle(len(rows), seed=shuffle_seed)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ELPV classifiers.")
    parser.add_argument("--quick", action="store_true", help="Run one epoch of A using validation only.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per condition (default: 15).")
    parser.add_argument("--seed", type=int, default=42, help="Training and sampling seed (default: 42).")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    folder = Path(__file__).resolve().parent
    data_dir = folder / "elpv-dataset-master/src/elpv_dataset/data"
    with (folder / "splits.csv").open(encoding="utf-8") as stream:
        splits = list(csv.DictReader(stream))
    train = [row for row in splits if row["split"] == "train"]
    validation = image_batches([row for row in splits if row["split"] == "validation"], data_dir)
    normal = [row for row in train if row["binary_label"] == "0"]
    defective = [row for row in train if row["binary_label"] == "1"]
    random = Random(args.seed)
    random.shuffle(normal)
    random.shuffle(defective)
    conditions = {"A": CONDITIONS["A"]} if args.quick else CONDITIONS

    output_dir = folder / "results" / f"seed_{args.seed}"
    if not args.quick:
        output_dir.mkdir(parents=True, exist_ok=True)
        test = [row for row in splits if row["split"] == "test"]
        test_batches = image_batches(test, data_dir)
        actual = tf.constant([row["binary_label"] == "1" for row in test]).numpy()

    results = []
    predictions = []
    for condition, (normal_count, defective_count) in conditions.items():
        selected = normal[:normal_count] + defective[:defective_count]
        print(f"\nCondition {condition}: {normal_count} normal, {defective_count} defective", flush=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        model = build_model()
        model.fit(
            image_batches(selected, data_dir, shuffle_seed=args.seed),
            validation_data=validation,
            epochs=1 if args.quick else args.epochs,
            verbose=2,
        )
        if args.quick:
            print("Quick check completed using the validation set.")
            return

        scores = model.predict(test_batches, verbose=0).ravel()
        predicted = scores >= THRESHOLD
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        tn = int((~actual & ~predicted).sum())
        fn = int((actual & ~predicted).sum())
        results.append({
            "condition": condition, "seed": args.seed, "epochs": args.epochs,
            "train_normal": normal_count, "train_defect": defective_count,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(test),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        })
        predictions.extend({
            "condition": condition, "seed": args.seed, "image_path": row["image_path"],
            "true_label": row["binary_label"], "defect_score": float(score),
            "predicted_label": int(label),
        } for row, score, label in zip(test, scores, predicted, strict=True))
        model.save(output_dir / f"{condition}.keras")

    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "predictions.csv", predictions)
    print(f"\nSaved results.csv, predictions.csv, and trained models to {output_dir}")


if __name__ == "__main__":
    main()
