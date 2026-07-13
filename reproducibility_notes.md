# Reproducibility Notes

## Experimental Design

- Four-class classification: Monkeypox/Mpox, Chickenpox, Measles, Normal.
- Cleaned dataset size: 766 images.
- Split: 612 training, 77 validation, 77 test.
- The test set is held out and used once for final reporting.
- Model selection is based primarily on validation performance and final independent test evaluation.

## Models

The V2 experiments include six completed model variants:

- DenseNet201 with class-weighted cross-entropy
- DenseNet201 with focal loss
- EfficientNetB3 with class-weighted cross-entropy
- EfficientNetB3 with focal loss
- MobileNetV2 with class-weighted cross-entropy
- MobileNetV2 with focal loss

## Metrics

Reported metrics include accuracy, macro precision, macro recall, macro-F1, weighted-F1, negative log-likelihood, expected calibration error, and Brier score.

## Large Files

Raw images and trained `.weights.h5` files are excluded from the GitHub repository to keep the repository lightweight. Results are provided as CSV and figure outputs. Large artifacts can be hosted in Kaggle or another research repository.
