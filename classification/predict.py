import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from classification_models.ClassifierType import ClassifierType
from classification_models.CustomEfficientNet import CustomEfficientNet
from classification_models.CustomResnet import (
    CustomResnet18,
    CustomResnet50,
    CustomResnet101,
    CustomResnet152,
)


GENDER_ID_TO_LABEL = {
    0: "male",
    1: "female",
}

RACE_ID_TO_LABEL = {
    0: "white",
    1: "black",
    2: "asian",
    3: "indian",
    4: "other",
}


def get_base_model(model_config, num_classes=1):
    model_type = model_config.get("name", ClassifierType.RESNET18.value)
    pretrained = model_config.get("pretrained", False)
    start_from = model_config.get("train_start", None)
    custom_weights_path = model_config.get("custom_weights_path", None)
    type_efficientnet = model_config.get("type_efficientnet", "efficientnet-b0")

    if model_type == ClassifierType.RESNET18.value:
        return CustomResnet18(num_classes, start_from, pretrained, custom_weights_path)
    if model_type == ClassifierType.RESNET50.value:
        return CustomResnet50(num_classes, start_from, pretrained, custom_weights_path)
    if model_type == ClassifierType.RESNET101.value:
        return CustomResnet101(num_classes, start_from, pretrained, custom_weights_path)
    if model_type == ClassifierType.RESNET152.value:
        return CustomResnet152(num_classes, start_from, pretrained, custom_weights_path)
    if model_type == ClassifierType.EFFICIENTNET.value:
        return CustomEfficientNet(num_classes, start_from, pretrained, type_efficientnet)
    raise ValueError(f"Unsupported model type: {model_type}")


class MultiHeadClassifierWrapper(nn.Module):
    def __init__(self, base_classifier):
        super().__init__()
        self.base_classifier = base_classifier

        if hasattr(self.base_classifier.model, "fc"):
            in_features = self.base_classifier.model.fc.in_features
            self.base_classifier.model.fc = nn.Identity()
        elif hasattr(self.base_classifier.model, "_fc"):
            in_features = self.base_classifier.model._fc[-1].in_features
            self.base_classifier.model._fc[-1] = nn.Identity()
        else:
            raise NotImplementedError("Base model does not have a recognized final layer.")

        self.fc_gender = nn.Linear(in_features, len(GENDER_ID_TO_LABEL))
        self.fc_race = nn.Linear(in_features, len(RACE_ID_TO_LABEL))

    def forward(self, x):
        features = self.base_classifier.model(x)
        return self.fc_gender(features), self.fc_race(features)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Iterable[str], transform):
        self.image_paths = [str(path) for path in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            image = Image.open(img_path).convert("RGB")
            return {
                "index": index,
                "img_path": img_path,
                "image": self.transform(image),
                "error": None,
            }
        except Exception as exc:
            return {
                "index": index,
                "img_path": img_path,
                "image": None,
                "error": str(exc),
            }


def prediction_collate_fn(batch):
    valid = [item for item in batch if item["image"] is not None]
    errors = [item for item in batch if item["image"] is None]
    if valid:
        images = torch.stack([item["image"] for item in valid])
        indices = [item["index"] for item in valid]
        paths = [item["img_path"] for item in valid]
    else:
        images = None
        indices = []
        paths = []
    return {
        "images": images,
        "indices": indices,
        "paths": paths,
        "errors": errors,
    }


def load_yaml_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["__config_dir"] = str(config_path.parent)
    return config


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_path_from_run_id(
    run_id,
    classification_dir=THIS_DIR,
    filename="best_model.pt",
):
    return Path(classification_dir).expanduser().resolve() / "out" / run_id / filename


def resolve_checkpoint_path(checkpoint_path=None, config=None, classification_dir=THIS_DIR):
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute() and not path.exists():
            candidate = Path(classification_dir).expanduser().resolve() / path
            if candidate.exists():
                path = candidate
            elif config and config.get("__config_dir"):
                path = Path(config["__config_dir"]) / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    if config is None:
        raise ValueError("Pass checkpoint_path or config.")

    config_dir = Path(config.get("__config_dir", Path.cwd())).resolve()
    checkpoint_cfg = config.get("checkpoint", {})
    output_dir = Path(checkpoint_cfg.get("output_dir", "./out")).expanduser()
    if not output_dir.is_absolute():
        output_dir = config_dir / output_dir
    output_dir = output_dir.resolve()
    filename = checkpoint_cfg.get("best_model_filename", "best_model.pt")

    candidates = list(output_dir.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"No {filename} found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_prediction_model(
    checkpoint_path=None,
    run_id=None,
    config_path=None,
    classification_dir=THIS_DIR,
    device=None,
    channels_last=False,
):
    if checkpoint_path is not None and run_id is not None:
        raise ValueError("Pass checkpoint_path or run_id, not both.")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    config = None
    if run_id is not None:
        checkpoint_path = checkpoint_path_from_run_id(
            run_id=run_id,
            classification_dir=classification_dir,
        )
    elif checkpoint_path is not None:
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            classification_dir=classification_dir,
        )
    else:
        config = load_yaml_config(config_path or THIS_DIR / "classification-config.yaml")
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_path=None,
            config=config,
            classification_dir=classification_dir,
        )

    checkpoint = _torch_load(checkpoint_path, device)

    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if checkpoint_config:
        config = checkpoint_config
    elif config_path:
        config = load_yaml_config(config_path)
    elif config is None:
        raise ValueError(
            "This checkpoint does not contain a saved config. Pass config_path for this older checkpoint."
        )

    model_config = copy.deepcopy(config.get("model", {}))
    model_config["pretrained"] = False
    model_config["custom_weights_path"] = None
    base_classifier = get_base_model(model_config)
    transform = base_classifier.get_transform()
    model = MultiHeadClassifierWrapper(base_classifier).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    return model, transform, device, checkpoint_path


def clean_model_version(model_version):
    version = str(model_version).strip()
    if not version:
        raise ValueError("model_version must not be empty.")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in version)


@torch.inference_mode()
def predict_image_path(
    image_path,
    model,
    transform,
    device,
    channels_last=False,
):
    image_path = str(image_path)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    if channels_last and device.type == "cuda":
        image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)

    gender_logits, race_logits = model(image_tensor)
    gender_probs = torch.softmax(gender_logits, dim=1).squeeze(0).cpu()
    race_probs = torch.softmax(race_logits, dim=1).squeeze(0).cpu()

    gender_pred_id = int(torch.argmax(gender_probs).item())
    race_pred_id = int(torch.argmax(race_probs).item())

    return {
        "img_path": image_path,
        "gender_pred_id": gender_pred_id,
        "gender_pred_class": GENDER_ID_TO_LABEL[gender_pred_id],
        "gender_pred_probability": float(gender_probs[gender_pred_id].item()),
        "race_pred_id": race_pred_id,
        "race_pred_class": RACE_ID_TO_LABEL[race_pred_id],
        "race_pred_probability": float(race_probs[race_pred_id].item()),
        "gender_probabilities": {
            label: float(gender_probs[class_id].item())
            for class_id, label in GENDER_ID_TO_LABEL.items()
        },
        "race_probabilities": {
            label: float(race_probs[class_id].item())
            for class_id, label in RACE_ID_TO_LABEL.items()
        },
    }


def save_prediction_to_sample(
    sample,
    prediction,
    model_version,
    gender_field_prefix="gender_pred",
    race_field_prefix="race_pred",
):
    import fiftyone as fo

    model_version = clean_model_version(model_version)
    gender_field = f"{gender_field_prefix}_{model_version}"
    race_field = f"{race_field_prefix}_{model_version}"

    sample[gender_field] = fo.Classification(
        label=prediction["gender_pred_class"],
        confidence=float(prediction["gender_pred_probability"]),
    )
    sample[race_field] = fo.Classification(
        label=prediction["race_pred_class"],
        confidence=float(prediction["race_pred_probability"]),
    )
    sample.save()
    return gender_field, race_field


class ImagePathPredictor:
    def __init__(
        self,
        checkpoint_path=None,
        run_id=None,
        config_path=None,
        classification_dir=THIS_DIR,
        device=None,
        channels_last=False,
        model_version=None,
    ):
        self.model, self.transform, self.device, self.checkpoint_path = load_prediction_model(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            config_path=config_path,
            classification_dir=classification_dir,
            device=device,
            channels_last=channels_last,
        )
        self.channels_last = channels_last
        self.model_version = clean_model_version(
            model_version or self.checkpoint_path.parent.name
        )

    def predict(self, image_path):
        return predict_image_path(
            image_path=image_path,
            model=self.model,
            transform=self.transform,
            device=self.device,
            channels_last=self.channels_last,
        )

    def predict_sample(self, sample):
        return self.predict(sample.filepath)

    def save_sample_prediction(self, sample):
        prediction = self.predict_sample(sample)
        save_prediction_to_sample(
            sample=sample,
            prediction=prediction,
            model_version=self.model_version,
        )
        return prediction


@torch.inference_mode()
def predict_image_paths(
    image_paths,
    model,
    transform,
    device,
    batch_size=32,
    num_workers=0,
    channels_last=False,
):
    image_paths = [str(path) for path in image_paths]
    rows = [
        {
            "img_path": path,
            "prediction_error": None,
        }
        for path in image_paths
    ]

    dataset = ImagePathDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=prediction_collate_fn,
    )

    for batch in loader:
        for error_item in batch["errors"]:
            rows[error_item["index"]]["prediction_error"] = error_item["error"]

        images = batch["images"]
        if images is None:
            continue

        images = images.to(device)
        if channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        gender_logits, race_logits = model(images)
        gender_probs = torch.softmax(gender_logits, dim=1).cpu()
        race_probs = torch.softmax(race_logits, dim=1).cpu()
        gender_pred_ids = torch.argmax(gender_probs, dim=1)
        race_pred_ids = torch.argmax(race_probs, dim=1)

        for row_idx, gender_id, race_id, gender_prob_row, race_prob_row in zip(
            batch["indices"],
            gender_pred_ids.tolist(),
            race_pred_ids.tolist(),
            gender_probs.tolist(),
            race_probs.tolist(),
        ):
            rows[row_idx].update(
                {
                    "gender_pred_id": gender_id,
                    "gender_pred_class": GENDER_ID_TO_LABEL[gender_id],
                    "gender_pred_probability": gender_prob_row[gender_id],
                    "race_pred_id": race_id,
                    "race_pred_class": RACE_ID_TO_LABEL[race_id],
                    "race_pred_probability": race_prob_row[race_id],
                }
            )

            for class_id, label in GENDER_ID_TO_LABEL.items():
                rows[row_idx][f"gender_prob_{label}"] = gender_prob_row[class_id]
            for class_id, label in RACE_ID_TO_LABEL.items():
                rows[row_idx][f"race_prob_{label}"] = race_prob_row[class_id]

    return pd.DataFrame(rows)


def predict_dataframe(
    df,
    image_col="img_path",
    checkpoint_path=None,
    run_id=None,
    config_path=None,
    classification_dir=THIS_DIR,
    device=None,
    batch_size=32,
    num_workers=0,
    channels_last=False,
):
    model, transform, device, checkpoint_path = load_prediction_model(
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        config_path=config_path,
        classification_dir=classification_dir,
        device=device,
        channels_last=channels_last,
    )
    predictions = predict_image_paths(
        df[image_col].tolist(),
        model,
        transform,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        channels_last=channels_last,
    )
    predictions = predictions.drop(columns=["img_path"])
    result = pd.concat(
        [
            df.reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )
    result.attrs["checkpoint_path"] = str(checkpoint_path)
    return result


def attach_predictions_to_fiftyone_samples(
    samples,
    predictions_df,
    filepath_col="img_path",
    gender_field_prefix="gender_pred",
    race_field_prefix="race_pred",
    model_version=None,
):
    import fiftyone as fo

    if model_version is not None:
        model_version = clean_model_version(model_version)
        gender_field = f"{gender_field_prefix}_{model_version}"
        race_field = f"{race_field_prefix}_{model_version}"
    else:
        gender_field = gender_field_prefix
        race_field = race_field_prefix

    by_path = {
        str(row[filepath_col]): row
        for _, row in predictions_df.iterrows()
        if pd.isna(row.get("prediction_error"))
    }

    for sample in samples:
        row = by_path.get(str(sample.filepath))
        if row is None:
            continue
        sample[gender_field] = fo.Classification(
            label=str(row["gender_pred_class"]),
            confidence=float(row["gender_pred_probability"]),
        )
        sample[race_field] = fo.Classification(
            label=str(row["race_pred_class"]),
            confidence=float(row["race_pred_probability"]),
        )
        sample.save()


def main():
    parser = argparse.ArgumentParser(description="Run gender/race predictions.")
    parser.add_argument("--config", default=str(THIS_DIR / "classification-config.yaml"))
    parser.add_argument("--classification-dir", default=str(THIS_DIR))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--image-col", default="img_path")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--output-csv", default="predictions.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--channels-last", action="store_true")
    args = parser.parse_args()

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    elif args.image:
        df = pd.DataFrame({args.image_col: args.image})
    else:
        raise ValueError("Pass --input-csv or one/more --image values.")

    result = predict_dataframe(
        df,
        image_col=args.image_col,
        checkpoint_path=args.checkpoint,
        run_id=args.run_id,
        config_path=args.config,
        classification_dir=args.classification_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        channels_last=args.channels_last,
    )
    result.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {Path(args.output_csv).resolve()}")
    print(f"Checkpoint: {result.attrs['checkpoint_path']}")


if __name__ == "__main__":
    main()
