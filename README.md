# AU10TIX Computer Vision Assignment

This repository contains the review-ready code and notebooks for the AU10TIX computer vision assignment.

The work is split into two separate tasks:

1. **Face pose estimation from facial landmarks**
2. **Gender and race classification**

Each task uses a different dataset and is handled by different files in the repository.

## Tasks and Datasets

### 1. Face Pose Estimation

This task uses a small landmark dataset containing **48 face images**, each paired with an annotation file containing **5 facial landmarks**:

- left eye
- right eye
- nose
- left mouth corner
- right mouth corner

The landmark task checks the image and annotation pairs, explores the annotation quality, calculates face geometry features, and attaches the inferred values to samples in FiftyOne/Voxel51.

Main files:

- `landmarks.ipynb` - data checks, annotation verification, landmark exploration, face geometry calculation, and FiftyOne/Voxel51 enrichment.
- `face_position_calculation.md` - explanation of how the geometry attributes are calculated from the landmarks.

### 2. Gender and Race Classification

This task uses the **UTKFace dataset** for multi-head classification of gender and race.

The classification task includes dataset preparation, data exploration, model training, and prediction.

Main files:

- `report.pdf` - report for the assignment
- `classification/classification.py` - training script for the gender and race classifier.
- `classification/predict.py` - inference script for running predictions with a trained checkpoint.
- `classification/models` - directory contains networks.
- `classification/create_csv.py` - this helper script is only for creating a sample csv for testing the training script. The construction of the dataset for the real training is performed within the notebook. 
- `classification/utkface_v2.csv` - final UTKFace metadata CSV used by the default training configuration.
- `face_classification/utkf_data_explor_clean.ipynb` - notebook for exploration and cleaning dataset, dataset preparation, samples tagging and prediction.

## Repository Contents

- `landmarks.ipynb` - face landmark checks, exploration, geometry calculation, and FiftyOne/Voxel51 integration.
- `face_position_calculation.md` - readable explanation of the face geometry calculations.
- `classification/` - gender and race classification training, prediction, configuration, and model wrapper code.
- `classification/classification-config.yaml` - default classification training configuration and file to be adjusted to alter training parameters.
- `classification/utkface_v2.csv` - final UTKFace dataset metadata CSV kept for review.
- `face_classification/utkf_data_explor_clean.ipynb` - notebook for exploration and cleaning dataset, dataset preparation, samples tagging and prediction.
- `assets/face_rejected.png` - example visual asset used by the landmark notebook/documentation.

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

The default config uses `utkface_v2.csv`. Training requires the image paths referenced by the CSV to exist locally. Currently, the dataset contains absolute paths - need to be changed and potentially the classification scripts adapted.

## Predict

After training or receiving a checkpoint locally:

```bash
cd classification
python predict.py --checkpoint out/<run_id>/best_model.pt --image path\to\image.jpg --output-csv predictions.csv
```

Model checkpoints are excluded from the repository by design.
