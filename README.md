# EF2039 PA04 | Solar Cell Defect Detection

This project evaluates how training set size and class imbalance affect CNN-based defect classification on electroluminescence (EL) images of solar cells. All five conditions share the same 200 test images.

## 1. Download the ELPV dataset

Download the [ELPV dataset ZIP](https://github.com/zae-bayern/elpv-dataset/archive/refs/heads/master.zip) and save it as `elpv-dataset-master.zip` beside the Python scripts. The archive contains the EL images and their [original labels](https://github.com/zae-bayern/elpv-dataset/blob/master/src/elpv_dataset/data/labels.csv). See the [official ELPV repository](https://github.com/zae-bayern/elpv-dataset) for citations and licensing.

The binary task uses label 0 for normal cells and label 1 for defective cells. The intermediate labels are excluded. The fixed split reserves 50 images per class for validation and 100 per class for testing. Training examples are sampled from the remaining images.

## 2. Create the Conda environment

Run these commands from the project directory in a Linux terminal. The reported experiments ran on Backend.AI with Python 3.10.0 and TensorFlow 2.15.0. A CPU can run the training and evaluation scripts.

```bash
conda env create -n pa04 -f environment.yml
conda activate pa04
export PYTHONNOUSERSITE=1
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

If you already have the PA04 environment, activate it and check the versions. The `-n pa04` option gives the new environment a local name even if the export was generated under a different name.

## 3. Extract the data and prepare the split

If you have the original `splits.csv`, place it beside the scripts and extract the archive.

```bash
python -m zipfile -e elpv-dataset-master.zip .
```

If you need to generate `splits.csv`, run the preparation script. It extracts the archive as needed and creates the split with seed 42. If the environment is missing pandas, first run `python -m pip install "pandas==2.2.3"`.

```bash
python prepare_data.py
```

## 4. Train and evaluate

```bash
python train_elpv.py --quick
python train_elpv.py --epochs 15 --seed 42
```

The quick check trains condition A for one epoch and evaluates it on the validation set. The full run trains and evaluates the following conditions on the same held-out test set.

| Condition | Normal | Defective | Total | Comparison |
|---|---:|---:|---:|---|
| A | 20 | 20 | 40 | Training set size |
| B | 50 | 50 | 100 | Training set size |
| C | 100 | 100 | 200 | Both comparisons |
| E | 150 | 50 | 200 | Class imbalance |
| F | 180 | 20 | 200 | Class imbalance |

The full run writes `results.csv`, `predictions.csv`, and five `.keras` models to `results/seed_42/`. It replaces files with the same names in that directory. In the reported Conda run, defect recall for A, B, C, E, and F was 1%, 28%, 48%, 4%, and 0%, respectively. Repeated training can produce different scores.

## 5. Generate figures and inspect errors

```bash
python analyze_elpv.py --seed 42
python analyze_elpv_deep.py --seeds 42
python show_predictions.py --seed 42 --condition C --sample
```

- `results/seed_42/figure1_with_values.png` shows recall and F1 with the test metrics table.
- `results/deep_analysis/` contains condition summaries and paired changes in predictions for the same test cells.
- `results/seed_42/prediction_examples_C.png` shows eight original EL images with actual labels, predicted labels, and defect scores. Use `--all` in place of `--sample` to view all 200 images in pages of eight.
