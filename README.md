# AU10TIX Computer Vision Assignment

This repository contains the review-ready code and notebooks for the AU10TIX computer vision assignment. The main task is multi-head face classification for gender and race using UTKFace metadata.

## Repository Contents

- `classification/` - training, prediction, configuration, and model wrapper code.
- `classification/utkface_v2.csv` - final dataset metadata CSV kept for review.
- `classification/classification-config.yaml` - default training configuration.
- `face_classification/utkf_data_explor_clean.ipynb` - cleaned UTKFace exploration notebook.
- `data_explor_landmarks.ipynb` - landmark/data exploration notebook.

Raw image folders, sample data, assignment materials, MLflow runs, checkpoints, saved models, archives, and older CSV exports are intentionally excluded from Git.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
cd classification
python classification.py --config classification-config.yaml
```

The default config uses `utkface_v2.csv`. Training requires the image paths referenced by the CSV to exist locally.

## Predict

After training or receiving a checkpoint locally:

```bash
cd classification
python predict.py --checkpoint out/<run_id>/best_model.pt --image path\to\image.jpg --output-csv predictions.csv
```

Model checkpoints are excluded from the repository by design.
