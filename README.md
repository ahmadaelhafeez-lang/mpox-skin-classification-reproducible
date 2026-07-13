# Reproducible Mpox Skin Image Classification

This repository contains the reproducible code and result summaries for a four-class skin image classification experiment involving **Monkeypox/Mpox**, **Chickenpox**, **Measles**, and **Normal** skin images.

The workflow uses a cleaned image index, leakage-controlled train/validation/test split, transfer learning with convolutional neural network backbones, test-time augmentation, calibration metrics, and statistical analysis.

## Main Result

The best-performing model in the final V2 experiment was **MobileNetV2 trained with class-weighted cross-entropy**:

| Metric | Value |
|---|---:|
| Test accuracy | 92.21% |
| Macro-F1 | 92.39% |
| Macro precision | 93.58% |
| Macro recall | 91.44% |
| Weighted F1 | 92.26% |
| Expected calibration error | 0.053 |
| Brier score | 0.153 |
| Negative log-likelihood | 0.323 |

## Dataset Summary

The final cleaned dataset contained **766 images** distributed across four classes:

- Monkeypox/Mpox: 277
- Chickenpox: 106
- Measles: 90
- Normal: 293

The leakage-controlled split was:

- Training: 612 images
- Validation: 77 images
- Test: 77 images

## Repository Structure

```text
code/
  01_prepare_dataset.py
  02_train_resume_models.py
  03_final_analysis.py
  04_generate_figures_tables.py

results/
  csv/
  figures/

docs/
  reproducibility_notes.md
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts in order:

```bash
python code/01_prepare_dataset.py --data_root "/path/to/Monkeypox Skin Image Dataset" --output_dir outputs
python code/02_train_resume_models.py --data_root "/path/to/Monkeypox Skin Image Dataset" --output_dir outputs
python code/03_final_analysis.py --results_dir outputs/csv --output_dir outputs/final_analysis
python code/04_generate_figures_tables.py --results_dir outputs/csv --figures_dir outputs/figures --output_dir outputs/manuscript_tables
```

## Results

The `results/` folder includes the final CSV files and confusion-matrix figures generated from the V2 experiment. Large model-weight files and raw image files are intentionally excluded from this GitHub repository.

## Data Availability

The processed reproducibility outputs and/or result package should be made available through Kaggle. Add the public Kaggle link here after publishing the dataset:

```text
Kaggle dataset/results link: TO_BE_ADDED
```

## Code Availability

This repository is publicly available at:

```text
https://github.com/ahmadaelhafeez-lang/mpox-skin-classification-reproducible
```

## License

The code is released under the MIT License. Use of the original images must follow the terms of the original Monkeypox Skin Image Dataset source.
