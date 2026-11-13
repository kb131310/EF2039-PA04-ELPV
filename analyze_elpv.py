import argparse
import csv
from pathlib import Path


CONDITIONS = ("A", "B", "C", "E", "F")
METRICS = ("accuracy", "precision", "recall", "f1")


def print_results(results):
    print(f"{'Condition':<11}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'FN':>6}")
    for condition in CONDITIONS:
        row = results[condition]
        metrics = [float(row[name]) for name in METRICS]
        accuracy, precision, recall, f1 = metrics
        print(f"{condition:<11}{accuracy:>10.1%}{precision:>11.1%}{recall:>9.1%}{f1:>9.1%}{int(row['fn']):>6}")

    print("\nComparison: later condition minus earlier condition")
    for question, start, end in (("RQ1", "A", "B"), ("RQ1", "B", "C"),
                                  ("RQ2", "C", "E"), ("RQ2", "C", "F")):
        before, after = results[start], results[end]
        recall_change = (float(after["recall"]) - float(before["recall"])) * 100
        f1_change = (float(after["f1"]) - float(before["f1"])) * 100
        missed_change = int(after["fn"]) - int(before["fn"])
        print(f"{question} {start} -> {end}: Recall {recall_change:+.1f} pp, "
              f"F1 {f1_change:+.1f} pp, missed defects {missed_change:+d}")


def save_figure(results, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.875, 3.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(2, 1.25))
    size_axis = fig.add_subplot(grid[0, 0])
    ratio_axis = fig.add_subplot(grid[0, 1], sharey=size_axis)
    ratio_axis.tick_params(labelleft=False)
    table_axis = fig.add_subplot(grid[1, :])
    table_axis.set_axis_off()
    size_conditions = ("A", "B", "C")
    size_ticks = [int(results[c]["train_normal"]) + int(results[c]["train_defect"])
                  for c in size_conditions]
    ratio_conditions = ("C", "E", "F")
    ratio_ticks = [f"{int(results[c]['train_normal']) // int(results[c]['train_defect'])}:1"
                   for c in ratio_conditions]

    for axis, conditions, x in ((size_axis, size_conditions, size_ticks),
                                (ratio_axis, ratio_conditions, ratio_ticks)):
        recall = [100 * float(results[c]["recall"]) for c in conditions]
        f1 = [100 * float(results[c]["f1"]) for c in conditions]
        axis.plot(x, recall, "o-", color="#175782", markersize=4, label="Defect recall")
        axis.plot(x, f1, "s--", color="#a84511", markersize=7,
                  markerfacecolor="none", label="Defect F1")
        axis.set_ylim(-4, 104)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", alpha=0.25)

    size_axis.set_title("(a) Training data size", fontsize=10)
    size_axis.set_xlabel("Training images")
    size_axis.set_ylabel("Test score (%)")
    size_axis.set_xticks(size_ticks)
    size_axis.legend(loc="best", fontsize=8)
    ratio_axis.set_title("(b) Class imbalance", fontsize=10)
    ratio_axis.set_xlabel("Normal : defective training images")

    table_rows = []
    for condition in CONDITIONS:
        row = results[condition]
        table_rows.append([
            condition,
            f"{row['train_normal']} / {row['train_defect']}",
            *(f"{float(row[metric]):.1%}" for metric in METRICS),
            row["fn"],
        ])
    table = table_axis.table(
        cellText=table_rows,
        colLabels=("Condition", "Train N / D", "Accuracy", "Precision", "Recall", "F1", "FN (cells)"),
        colWidths=(0.13, 0.22, 0.14, 0.14, 0.13, 0.11, 0.13),
        cellLoc="center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    baseline_row = CONDITIONS.index("C") + 1
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c1c7cc")
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_facecolor("#eef0f2")
            cell.set_text_props(weight="bold")
        elif row_index == baseline_row:
            cell.set_facecolor("#e9f1f7")

    fig.savefig(destination, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot an ELPV experiment run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for training (default: 42).")
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent / "results" / f"seed_{args.seed}"
    results_path = folder / "results.csv"
    if not results_path.is_file():
        parser.error(f"Results file not found: {results_path}. Run full training first.")

    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    results = {row["condition"]: row for row in rows}
    if len(rows) != len(CONDITIONS) or set(results) != set(CONDITIONS):
        parser.error("results.csv must have one row each for A, B, C, E, and F.")

    print_results(results)
    destination = folder / "figure1_with_values.png"
    save_figure(results, destination)
    print(f"\nSaved {destination}")


if __name__ == "__main__":
    main()
