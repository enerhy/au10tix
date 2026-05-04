import os
import random
import re
import pandas as pd
from pathlib import Path


GENDER_TO_ID = {
    "male": 0,
    "female": 1,
}

RACE_TO_ID = {
    "white": 0,
    "black": 1,
    "asian": 2,
    "indian": 3,
    "other": 4,
}

SPLIT_TO_ID = {
    "train": 0,
    "validation": 1,
    "val": 1,
    "test": 2,
}


def generate_csv(data_dir, output_csv):
    data = []
    
    # ensure deterministic splitting
    random.seed(42)
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            parts = filename.split('_')
            if len(parts) >= 4:
                age = parts[0]
                gender = parts[1]
                race = parts[2]
                
                # randomly assign split: 0=train (70%), 1=val (15%), 2=test (15%)
                rand_val = random.random()
                if rand_val < 0.7:
                    split = 0
                elif rand_val < 0.85:
                    split = 1
                else:
                    split = 2
                    
                row = {
                    'img_path': str(Path(data_dir) / filename),
                    'gender': int(gender),
                    'race': int(race),
                    'split': split,
                    'other_attributes': '{}'
                }
                data.append(row)
                
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated {output_csv} with {len(df)} rows.")


def versioned_csv_path(output_csv, version):
    output_csv = Path(output_csv)

    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv.with_suffix(".csv")

    base_stem = re.sub(
        r"[_-]v(?:ersion)?[_-]?[0-9]+(?:\.[0-9]+)*$",
        "",
        output_csv.stem,
        flags=re.IGNORECASE,
    )
    output_csv = output_csv.with_name(f"{base_stem}_v{version}{output_csv.suffix}")

    return output_csv


def export_fiftyone_dataset_to_csv(
    dataset,
    output_csv,
    version=1,
    gender_field="gender",
    race_field="race",
    split_tags=("train", "validation", "test"),
    duplicate_tag="duplicate_clip",
):
    """
    Exports a FiftyOne dataset to the classification CSV format:

    img_path, gender, race, split, other_attributes

    Split is read from sample tags. Samples without a split tag are exported
    with an empty split value, so the training script can ignore them.
    Samples tagged as duplicate_clip are skipped.

    - train -> 0
    - validation -> 1
    - test -> 2
    """

    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm

    rows = []
    skipped_duplicates = 0
    skipped_unknown_class = 0
    skipped_multiple_splits = 0
    no_split_count = 0
    split_tag_set = set(split_tags)

    for sample in tqdm(dataset, total=len(dataset), desc="Exporting dataset to CSV"):
        if duplicate_tag in sample.tags:
            skipped_duplicates += 1
            continue

        gender = sample[gender_field]
        race = sample[race_field]

        if gender not in GENDER_TO_ID or race not in RACE_TO_ID:
            skipped_unknown_class += 1
            continue

        sample_split_tags = [tag for tag in sample.tags if tag in split_tag_set]

        if len(sample_split_tags) > 1:
            skipped_multiple_splits += 1
            continue

        split = ""
        if sample_split_tags:
            split = SPLIT_TO_ID[sample_split_tags[0]]
        else:
            no_split_count += 1

        rows.append(
            {
                "img_path": sample.filepath,
                "gender": GENDER_TO_ID[gender],
                "race": RACE_TO_ID[race],
                "split": split,
                "other_attributes": "{}",
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "img_path",
            "gender",
            "race",
            "split",
            "other_attributes",
        ],
    )
    output_csv = versioned_csv_path(output_csv, version)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Generated {output_csv} with {len(df)} rows.")
    print(f"Skipped duplicates: {skipped_duplicates}")
    print(f"Skipped unknown race/gender: {skipped_unknown_class}")
    print(f"Skipped multiple split tags: {skipped_multiple_splits}")
    print(f"Rows with empty split: {no_split_count}")
    return df

if __name__ == '__main__':
    data_dir = 'sample_data'
    output_csv = 'data.csv'
    generate_csv(data_dir, output_csv)
