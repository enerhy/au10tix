import yaml
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import logging
import copy
import hashlib
import json
import re
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import mlflow
import mlflow.pytorch

from classification_models.ClassifierType import ClassifierType
from classification_models.CustomResnet import CustomResnet18, CustomResnet50, CustomResnet101, CustomResnet152
from classification_models.CustomEfficientNet import CustomEfficientNet

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GENDER_CLASSES = 2
RACE_CLASSES = 5

# ----------------- Base Model factory -----------------
def get_base_model(model_config, num_classes=1):
    model_type_str = model_config.get('name', 'Resnet18')
    pretrained = model_config.get('pretrained', True)
    start_from = model_config.get('train_start', None)
    custom_weights_path = model_config.get('custom_weights_path', None)
    type_efficientnet = model_config.get('type_efficientnet', 'efficientnet-b0')

    if model_type_str == ClassifierType.RESNET18.value:
        return CustomResnet18(num_classes, start_from, pretrained, custom_weights_path)
    elif model_type_str == ClassifierType.RESNET50.value:
        return CustomResnet50(num_classes, start_from, pretrained, custom_weights_path)
    elif model_type_str == ClassifierType.RESNET101.value:
        return CustomResnet101(num_classes, start_from, pretrained, custom_weights_path)
    elif model_type_str == ClassifierType.RESNET152.value:
        return CustomResnet152(num_classes, start_from, pretrained, custom_weights_path)
    elif model_type_str == ClassifierType.EFFICIENTNET.value:
        return CustomEfficientNet(num_classes, start_from, pretrained, type_efficientnet)
    else:
        raise ValueError(f"Unsupported model type: {model_type_str}")

# ----------------- Models -----------------

class MultiHeadClassifierWrapper(nn.Module):
    def __init__(self, base_classifier):
        super(MultiHeadClassifierWrapper, self).__init__()
        self.base_classifier = base_classifier
        
        # Replace the fully connected layer with Identity to extract features
        if hasattr(self.base_classifier.model, 'fc'):
            in_features = self.base_classifier.model.fc.in_features
            self.base_classifier.model.fc = nn.Identity()
        elif hasattr(self.base_classifier.model, '_fc'):
            # For CustomEfficientNet, _fc is a Sequential block, we replace the last layer
            in_features = self.base_classifier.model._fc[-1].in_features
            self.base_classifier.model._fc[-1] = nn.Identity()
        else:
            raise NotImplementedError("Base model does not have a recognized final layer ('fc' or '_fc').")
            
        # Gender classification head (2 classes)
        self.fc_gender = nn.Linear(in_features, 2)
        # Race classification head (5 classes)
        self.fc_race = nn.Linear(in_features, 5)
        
    def forward(self, x):
        features = self.base_classifier.model(x)
        gender_preds = self.fc_gender(features)
        race_preds = self.fc_race(features)
        return gender_preds, race_preds

    @property
    def params_to_update(self):
        params = self.base_classifier.params_to_update
        params.extend(list(self.fc_gender.parameters()))
        params.extend(list(self.fc_race.parameters()))
        return params


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, weight_gender=None, weight_race=None):
        super(MultiTaskLossWrapper, self).__init__()
        self.log_var_gender = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_race = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.criterion_gender = nn.CrossEntropyLoss(weight=weight_gender)
        self.criterion_race = nn.CrossEntropyLoss(weight=weight_race)

    def forward(self, preds_gender, preds_race, targets_gender, targets_race):
        loss_gender = self.criterion_gender(preds_gender, targets_gender)
        loss_race = self.criterion_race(preds_race, targets_race)
        
        precision_gender = torch.exp(-self.log_var_gender)
        loss_gender_weighted = precision_gender * loss_gender + self.log_var_gender
        
        precision_race = torch.exp(-self.log_var_race)
        loss_race_weighted = precision_race * loss_race + self.log_var_race
        
        return loss_gender_weighted + loss_race_weighted, loss_gender, loss_race


class PaperUncertaintyMultiTaskCELoss(nn.Module):
    def __init__(self, weight_gender=None, weight_race=None):
        super(PaperUncertaintyMultiTaskCELoss, self).__init__()
        self.log_var_gender = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.log_var_race = nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.criterion_gender = nn.CrossEntropyLoss(weight=weight_gender)
        self.criterion_race = nn.CrossEntropyLoss(weight=weight_race)

    def forward(self, preds_gender, preds_race, targets_gender, targets_race):
        loss_gender = self.criterion_gender(preds_gender, targets_gender)
        loss_race = self.criterion_race(preds_race, targets_race)

        precision_gender = torch.exp(-self.log_var_gender)
        loss_gender_weighted = precision_gender * loss_gender + 0.5 * self.log_var_gender

        precision_race = torch.exp(-self.log_var_race)
        loss_race_weighted = precision_race * loss_race + 0.5 * self.log_var_race

        return loss_gender_weighted + loss_race_weighted, loss_gender, loss_race


class SummedMultiTaskCELoss(nn.Module):
    def __init__(self, weight_gender=None, weight_race=None):
        super(SummedMultiTaskCELoss, self).__init__()
        self.criterion_gender = nn.CrossEntropyLoss(weight=weight_gender)
        self.criterion_race = nn.CrossEntropyLoss(weight=weight_race)

    def forward(self, preds_gender, preds_race, targets_gender, targets_race):
        loss_gender = self.criterion_gender(preds_gender, targets_gender)
        loss_race = self.criterion_race(preds_race, targets_race)

        return loss_gender + loss_race, loss_gender, loss_race


class FixedWeightedMultiTaskCELoss(nn.Module):
    def __init__(self, weight_gender=None, weight_race=None, task_weight_gender=1.0, task_weight_race=1.0):
        super(FixedWeightedMultiTaskCELoss, self).__init__()
        self.task_weight_gender = float(task_weight_gender)
        self.task_weight_race = float(task_weight_race)
        self.criterion_gender = nn.CrossEntropyLoss(weight=weight_gender)
        self.criterion_race = nn.CrossEntropyLoss(weight=weight_race)

    def forward(self, preds_gender, preds_race, targets_gender, targets_race):
        loss_gender = self.criterion_gender(preds_gender, targets_gender)
        loss_race = self.criterion_race(preds_race, targets_race)

        combined_loss = self.task_weight_gender * loss_gender + self.task_weight_race * loss_race
        return combined_loss, loss_gender, loss_race


# ----------------- Dataset -----------------

class FaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'img_path']
        gender = int(self.df.loc[idx, 'gender'])
        race = int(self.df.loc[idx, 'race'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error reading image {img_path}: {e}")
            return None, None
            
        if self.transform:
            image = self.transform(image)
            
        return image, {'gender': gender, 'race': race}

def custom_collate_fn(batch):
    batch = [b for b in batch if b[0] is not None and b[1] is not None]
    if len(batch) == 0:
        return None, None
    images, labels = zip(*batch)
    images = torch.stack(images)
    gender_labels = torch.tensor([lbl['gender'] for lbl in labels], dtype=torch.long)
    race_labels = torch.tensor([lbl['race'] for lbl in labels], dtype=torch.long)
    return images, {'gender': gender_labels, 'race': race_labels}


def compute_class_weights(df, column_name, num_classes):
    counts = (
        df[column_name]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .sort_index()
    )
    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        raise ValueError(
            f"Training split is missing {column_name} class(es): {missing}. "
            "Cannot compute stable class weights."
        )
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def build_loss_function(loss_cfg, train_df, device):
    loss_type = str(loss_cfg.get('type', 'OriginalUncertaintyMultiTaskLoss')).strip()
    legacy_weighted_ce = loss_type == 'WeightedCrossEntropy'
    use_class_weights = bool(loss_cfg.get('class_weights', legacy_weighted_ce))

    weight_gender = None
    weight_race = None
    if use_class_weights:
        logger.info("Using class weights inside the task cross-entropy losses")
        weight_gender = compute_class_weights(train_df, 'gender', GENDER_CLASSES).to(device)
        weight_race = compute_class_weights(train_df, 'race', RACE_CLASSES).to(device)

    if legacy_weighted_ce:
        logger.info(
            "loss.type=WeightedCrossEntropy is kept as a legacy alias for "
            "OriginalUncertaintyMultiTaskLoss with class_weights=True"
        )
        return MultiTaskLossWrapper(weight_gender=weight_gender, weight_race=weight_race)

    if loss_type in ('OriginalUncertaintyMultiTaskLoss', 'MultiTaskLossWrapper'):
        return MultiTaskLossWrapper(weight_gender=weight_gender, weight_race=weight_race)

    if loss_type == 'PaperUncertaintyMultiTaskCELoss':
        return PaperUncertaintyMultiTaskCELoss(weight_gender=weight_gender, weight_race=weight_race)

    if loss_type == 'SummedMultiTaskCELoss':
        return SummedMultiTaskCELoss(weight_gender=weight_gender, weight_race=weight_race)

    if loss_type == 'FixedWeightedMultiTaskCELoss':
        task_weights = loss_cfg.get('task_weights', {})
        task_weight_gender = task_weights.get('gender', loss_cfg.get('task_weight_gender', 1.0))
        task_weight_race = task_weights.get('race', loss_cfg.get('task_weight_race', 1.0))
        return FixedWeightedMultiTaskCELoss(
            weight_gender=weight_gender,
            weight_race=weight_race,
            task_weight_gender=task_weight_gender,
            task_weight_race=task_weight_race,
        )

    supported_types = [
        'OriginalUncertaintyMultiTaskLoss',
        'PaperUncertaintyMultiTaskCELoss',
        'SummedMultiTaskCELoss',
        'FixedWeightedMultiTaskCELoss',
        'WeightedCrossEntropy',
    ]
    raise ValueError(f"Unsupported loss.type '{loss_type}'. Supported values: {supported_types}")


def resolve_path(path_value, base_dir, must_exist=False):
    if path_value is None or str(path_value).strip() == '':
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def resolve_sqlite_tracking_uri(uri, base_dir):
    if not uri:
        uri = "sqlite:///mlflow.db"
    sqlite_prefix = "sqlite:///"
    if uri.startswith(sqlite_prefix):
        db_path = uri[len(sqlite_prefix):]
        if db_path and not Path(db_path).is_absolute():
            return f"{sqlite_prefix}{(base_dir / db_path).resolve().as_posix()}"
    return uri


def flatten_dict(data, prefix=''):
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value)
        else:
            flat[name] = value
    return flat


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_version_from_csv_path(csv_path):
    stem = Path(csv_path).stem
    match = re.search(r'(?:^|[_-])v(?:ersion)?[_-]?([0-9]+(?:\.[0-9]+)*)$', stem, re.IGNORECASE)
    if match:
        return match.group(1)
    return 'unknown'


def split_counts(df):
    split_name = {0: 'train', 1: 'val', 2: 'test'}
    counts = {}
    for split_value, count in df['split'].value_counts(dropna=False).items():
        key = split_name.get(split_value, str(split_value))
        counts[key] = int(count)
    return counts


def label_distribution(df, column_name, num_classes):
    counts = (
        df[column_name]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .sort_index()
    )
    return {str(k): int(v) for k, v in counts.items()}

# ----------------- Transforms & Augmentations -----------------

TRANFORM_DICT = {
    'horizontal_flip': lambda x: transforms.RandomHorizontalFlip(**x),
    'vertical_flip': lambda x: transforms.RandomVerticalFlip(**x),
    'color_jiter': lambda x: transforms.ColorJitter(**x),
    'random_rotation': lambda x: transforms.RandomRotation(**x),
    'random_affine': lambda x: transforms.RandomAffine(**x)
}

def augment_transform(
        augment: dict, transform: transforms.Compose) -> transforms.Compose:
    '''
    Add augmentations to the Compose transformation
    Args:
        augment (dict): dict from the config file defining the transformation
        transform (transforms.Compose): Transformation as per Network Class
    Return:
        transform (transforms.Compose): augmented Compose transformations
    '''

    augment_transform_compose = [
        y(augment[x]) for x, y in TRANFORM_DICT.items() if x in augment.keys()]

    for i in augment_transform_compose: 
        transform.transforms.insert(-2, i)
    return transform


def update_confusion_matrix(confusion, preds, targets):
    for pred, target in zip(preds.view(-1), targets.view(-1)):
        confusion[int(target), int(pred)] += 1


def metrics_from_confusion(confusion, prefix):
    confusion = confusion.to(dtype=torch.float64)
    total = confusion.sum().item()
    correct = confusion.diag().sum().item()

    precision = confusion.diag() / confusion.sum(dim=0).clamp(min=1)
    recall = confusion.diag() / confusion.sum(dim=1).clamp(min=1)
    f1 = (2 * precision * recall) / (precision + recall).clamp(min=1e-12)

    metrics = {
        f'{prefix}_acc': correct / total if total else 0.0,
        f'{prefix}_balanced_acc': recall.mean().item(),
        f'{prefix}_macro_precision': precision.mean().item(),
        f'{prefix}_macro_recall': recall.mean().item(),
        f'{prefix}_macro_f1': f1.mean().item(),
    }

    for class_idx in range(confusion.shape[0]):
        metrics[f'{prefix}_class_{class_idx}_precision'] = precision[class_idx].item()
        metrics[f'{prefix}_class_{class_idx}_recall'] = recall[class_idx].item()
        metrics[f'{prefix}_class_{class_idx}_f1'] = f1[class_idx].item()

    return metrics


def save_confusion_artifacts(confusions, artifact_prefix):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for head, confusion in confusions.items():
            cm_path = tmpdir / f'{artifact_prefix}_{head}_confusion_matrix.csv'
            pd.DataFrame(confusion.cpu().numpy()).to_csv(cm_path, index=True)
            mlflow.log_artifact(str(cm_path), artifact_path='confusion_matrices')


def run_phase(
    model,
    criterion,
    optimizer,
    loader,
    phase,
    device,
    scaler=None,
    use_amp=False,
    channels_last=False,
):
    is_train = phase == 'train'
    if is_train:
        model.train()
        criterion.train()
    else:
        model.eval()
        criterion.eval()

    running_loss = 0.0
    running_loss_g = 0.0
    running_loss_r = 0.0
    total_samples = 0
    confusions = {
        'gender': torch.zeros(GENDER_CLASSES, GENDER_CLASSES, dtype=torch.long),
        'race': torch.zeros(RACE_CLASSES, RACE_CLASSES, dtype=torch.long),
    }

    for images, labels in tqdm(loader, desc=f"{phase}"):
        if images is None:
            continue
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        targets_gender = labels['gender'].to(device, non_blocking=True)
        targets_race = labels['race'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds_gender, preds_race = model(images)
                loss, loss_g, loss_r = criterion(
                    preds_gender,
                    preds_race,
                    targets_gender,
                    targets_race,
                )

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_loss_g += loss_g.item() * batch_size
        running_loss_r += loss_r.item() * batch_size
        total_samples += batch_size

        g_preds = torch.argmax(preds_gender.detach(), dim=1).cpu()
        r_preds = torch.argmax(preds_race.detach(), dim=1).cpu()
        update_confusion_matrix(confusions['gender'], g_preds, targets_gender.cpu())
        update_confusion_matrix(confusions['race'], r_preds, targets_race.cpu())

    if total_samples == 0:
        raise ValueError(f"No valid samples were processed for phase '{phase}'.")

    metrics = {
        f'{phase}_loss': running_loss / total_samples,
        f'{phase}_gender_loss': running_loss_g / total_samples,
        f'{phase}_race_loss': running_loss_r / total_samples,
    }
    metrics.update(metrics_from_confusion(confusions['gender'], f'{phase}_gender'))
    metrics.update(metrics_from_confusion(confusions['race'], f'{phase}_race'))

    return metrics, confusions


def log_phase_summary(phase, metrics):
    log_msg = (
        f"{phase.capitalize()} | Combined Loss: {metrics[f'{phase}_loss']:.4f} | "
        f"Gender Loss: {metrics[f'{phase}_gender_loss']:.4f} | "
        f"Race Loss: {metrics[f'{phase}_race_loss']:.4f} | "
        f"Gender Acc: {metrics[f'{phase}_gender_acc']:.4f} | "
        f"Race Acc: {metrics[f'{phase}_race_acc']:.4f} | "
        f"Gender Macro F1: {metrics[f'{phase}_gender_macro_f1']:.4f} | "
        f"Race Macro F1: {metrics[f'{phase}_race_macro_f1']:.4f}"
    )
    logger.info(log_msg)
    print(log_msg)


# ----------------- Training Logic -----------------

def train_model(config, device):
    config_dir = Path(config.get('__config_dir', Path.cwd())).resolve()
    csv_path = resolve_path(config['csv_path'], config_dir, must_exist=True)
    epochs = int(config.get('epochs', 5))
    batch_size = int(config.get('batch_size', 32))
    training_cfg = config.get('training', {})
    dataloader_cfg = config.get('dataloader', {})
    num_workers = int(dataloader_cfg.get('num_workers', config.get('num_workers', 0)))
    pin_memory = bool(dataloader_cfg.get('pin_memory', device.type == 'cuda'))
    persistent_workers = bool(dataloader_cfg.get('persistent_workers', num_workers > 0))
    prefetch_factor = dataloader_cfg.get('prefetch_factor', 2)
    use_amp = bool(training_cfg.get('amp', device.type == 'cuda')) and device.type == 'cuda'
    channels_last = bool(training_cfg.get('channels_last', False)) and device.type == 'cuda'
    if bool(training_cfg.get('cudnn_benchmark', device.type == 'cuda')) and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model_config = config.get('model', {})
    checkpoint_cfg = config.get('checkpoint', {})

    output_dir = resolve_path(checkpoint_cfg.get('output_dir', './out'), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model setup
    base_classifier = get_base_model(model_config)
    model = MultiHeadClassifierWrapper(base_classifier).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Data split
    df = pd.read_csv(csv_path)
    if config.get('subset'):
        df = df.sample(int(config['subset']), random_state=42).reset_index(drop=True)

    train_df = df[df['split'] == 0].copy()
    val_df = df[df['split'] == 1].copy()
    test_df = df[df['split'] == 2].copy()

    if train_df.empty:
        raise ValueError("Training split is empty. Expected rows with split == 0.")
    if val_df.empty:
        raise ValueError("Validation split is empty. Expected rows with split == 1.")

    # Apply augmentations from config without constructing extra models.
    augment_cfg = config.get('augment', {})
    base_transform = base_classifier.get_transform()
    train_transform = copy.deepcopy(base_transform)
    if augment_cfg.get('train', False):
        train_transform = augment_transform(augment_cfg, train_transform)

    val_transform = copy.deepcopy(base_transform)
    if augment_cfg.get('valid', False) or augment_cfg.get('val', False):
        val_transform = augment_transform(augment_cfg, val_transform)

    test_transform = copy.deepcopy(base_transform)
    if augment_cfg.get('test', False):
        test_transform = augment_transform(augment_cfg, test_transform)

    train_dataset = FaceDataset(train_df, transform=train_transform)
    val_dataset = FaceDataset(val_df, transform=val_transform)
    test_dataset = FaceDataset(test_df, transform=test_transform) if not test_df.empty else None

    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': custom_collate_fn,
    }
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = persistent_workers
        dataloader_kwargs['prefetch_factor'] = int(prefetch_factor)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **dataloader_kwargs,
        )

    logger.info(
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, "
        f"Test size: {len(test_dataset) if test_dataset is not None else 0}"
    )

    # Loss setup
    loss_cfg = config.get('loss', {})
    criterion = build_loss_function(loss_cfg, train_df, device).to(device)
    logger.info(f"Using {criterion.__class__.__name__} loss")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Optimizer
    opt_cfg = config.get('optimizer', {})
    lr = float(opt_cfg.get('learning_rate', 1e-3))
    params_list = [{'params': model.params_to_update}]
    criterion_params = list(criterion.parameters())
    if criterion_params:
        params_list.append({'params': criterion_params, 'lr': lr})

    if opt_cfg.get('type', 'Adam') == 'Adam':
        optimizer = torch.optim.Adam(params_list, lr=lr)
    else:
        momentum = float(opt_cfg.get('momentum', 0.9))
        optimizer = torch.optim.SGD(params_list, lr=lr, momentum=momentum)

    # Scheduler setup. Keep typo aliases for older config files.
    scheduler = None
    scheduler_type = opt_cfg.get('lr_scheduler_type', opt_cfg.get('lr_schduler_type', None))
    if scheduler_type == 'reduce-on-plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=int(opt_cfg.get('lr_scheduler_patience', 3)),
        )
    elif scheduler_type == 'stepwise':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(opt_cfg.get('lr_scheduler_step_size', 5)),
            gamma=float(opt_cfg.get('lr_scheduler_gamma', 0.1)),
        )

    start_epoch = 1
    best_val_loss = float('inf')
    previous_best_val_loss = None
    resume_from = checkpoint_cfg.get('resume_from', config.get('resume_from', None))
    resume_from = resolve_path(resume_from, config_dir, must_exist=True)
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=bool(checkpoint_cfg.get('strict_load', True)),
        )
        if 'loss_wrapper_state_dict' in checkpoint:
            criterion.load_state_dict(checkpoint['loss_wrapper_state_dict'])
        if checkpoint_cfg.get('load_optimizer', True) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        if scheduler and checkpoint_cfg.get('load_scheduler', True) and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if use_amp and checkpoint_cfg.get('load_scaler', True) and 'scaler_state_dict' in checkpoint:
            scaler_state_dict = checkpoint['scaler_state_dict']
            if scaler_state_dict is not None:
                scaler.load_state_dict(scaler_state_dict)
        if checkpoint_cfg.get('continue_epoch_count', True):
            start_epoch = int(checkpoint.get('epoch', 0)) + 1
        previous_best_val_loss = float(checkpoint.get('best_val_loss', checkpoint.get('loss', float('inf'))))
        logger.info(f"Resumed checkpoint from {resume_from} at epoch {start_epoch}")

    if start_epoch > epochs:
        raise ValueError(
            f"start_epoch ({start_epoch}) is greater than configured epochs ({epochs}). "
            "Increase epochs when continuing training, or set checkpoint.continue_epoch_count to False."
        )

    mlflow_cfg = config.get('mlflow', {})
    experiment_name = mlflow_cfg.get('experiment_name', 'multihead_classification')
    tracking_uri = resolve_sqlite_tracking_uri(mlflow_cfg.get('tracking_uri'), config_dir)
    artifact_location = resolve_path(mlflow_cfg.get('artifact_location', './mlruns'), config_dir)
    artifact_location.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location.as_uri())
    mlflow.set_experiment(experiment_name)

    dataset_cfg = config.get('dataset', {})
    dataset_version = dataset_version_from_csv_path(csv_path)
    config_for_logging = {k: v for k, v in config.items() if not k.startswith('__')}

    with mlflow.start_run(run_name=mlflow_cfg.get('run_name')):
        run_id = mlflow.active_run().info.run_id
        run_output_dir = output_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = run_output_dir / checkpoint_cfg.get('best_model_filename', 'best_model.pt')

        mlflow.set_tags({
            'model_name': str(model_config.get('name', 'Resnet18')),
            'dataset_version': str(dataset_version),
            'task': 'multihead_gender_race_classification',
        })
        mlflow.log_params(flatten_dict(config_for_logging, prefix='config'))
        mlflow.log_params({
            'data.csv_path_abs': str(csv_path),
            'data.csv_sha256': sha256_file(csv_path),
            'dataset.version': dataset_version,
            'dataset.name': dataset_cfg.get('name', 'unknown'),
            'dataset.source': dataset_cfg.get('source', 'unknown'),
            'rows_total': int(len(df)),
            'rows_train': int(len(train_df)),
            'rows_val': int(len(val_df)),
            'rows_test': int(len(test_df)),
            'split_counts': json.dumps(split_counts(df), sort_keys=True),
            'train_gender_distribution': json.dumps(label_distribution(train_df, 'gender', GENDER_CLASSES), sort_keys=True),
            'train_race_distribution': json.dumps(label_distribution(train_df, 'race', RACE_CLASSES), sort_keys=True),
            'val_gender_distribution': json.dumps(label_distribution(val_df, 'gender', GENDER_CLASSES), sort_keys=True),
            'val_race_distribution': json.dumps(label_distribution(val_df, 'race', RACE_CLASSES), sort_keys=True),
            'test_gender_distribution': json.dumps(label_distribution(test_df, 'gender', GENDER_CLASSES), sort_keys=True),
            'test_race_distribution': json.dumps(label_distribution(test_df, 'race', RACE_CLASSES), sort_keys=True),
            'resume_from': str(resume_from) if resume_from else '',
            'resume_previous_best_val_loss': previous_best_val_loss if previous_best_val_loss is not None else '',
            'run_output_dir': str(run_output_dir),
        })
        mlflow.log_text(yaml.safe_dump(config_for_logging, sort_keys=False), 'config/classification-config.yaml')
        mlflow.log_artifact(str(csv_path), artifact_path='data')

        for epoch in range(start_epoch, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")

            train_metrics, _ = run_phase(
                model,
                criterion,
                optimizer,
                train_loader,
                'train',
                device,
                scaler=scaler,
                use_amp=use_amp,
                channels_last=channels_last,
            )
            log_phase_summary('train', train_metrics)
            mlflow.log_metrics(train_metrics, step=epoch)

            val_metrics, val_confusions = run_phase(
                model,
                criterion,
                optimizer,
                val_loader,
                'val',
                device,
                scaler=scaler,
                use_amp=use_amp,
                channels_last=channels_last,
            )
            log_phase_summary('val', val_metrics)
            mlflow.log_metrics(val_metrics, step=epoch)

            val_loss = val_metrics['val_loss']
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if use_amp else None,
                    'loss_wrapper_state_dict': criterion.state_dict(),
                    'loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'previous_best_val_loss': previous_best_val_loss,
                    'config': config_for_logging,
                    'dataset_version': dataset_version,
                    'csv_sha256': sha256_file(csv_path),
                    'mlflow_run_id': run_id,
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f"Saved new best model to {best_model_path}")
                print(f"--> Saved new best model to {best_model_path}")
                mlflow.log_artifact(str(best_model_path), "best_model")
                save_confusion_artifacts(val_confusions, f'epoch_{epoch}_val_best')

            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            lr_msg = f"Epoch {epoch} complete | Current LR: {current_lr:.6f}"
            logger.info(lr_msg)
            print(lr_msg)
            print("-" * 50)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        if best_model_path.exists():
            best_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            criterion.load_state_dict(best_checkpoint['loss_wrapper_state_dict'])
            mlflow.log_param('best_epoch', int(best_checkpoint['epoch']))
            mlflow.log_metric('best_val_loss', float(best_checkpoint['best_val_loss']))

            if test_loader is not None:
                test_metrics, test_confusions = run_phase(
                    model,
                    criterion,
                    optimizer,
                    test_loader,
                    'test',
                    device,
                    scaler=scaler,
                    use_amp=use_amp,
                    channels_last=channels_last,
                )
                log_phase_summary('test', test_metrics)
                mlflow.log_metrics(test_metrics)
                save_confusion_artifacts(test_confusions, 'test_best_model')

            if bool(mlflow_cfg.get('log_pytorch_model', False)):
                mlflow.pytorch.log_model(model, artifact_path='best_model_mlflow')
        else:
            logger.warning("No best model was saved; skipping MLflow model logging and test evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='classification-config.yaml', help='Path to config file')
    parser.add_argument('--subset', type=int, default=None, help='Subset of data to use for testing')
    args = parser.parse_args()
    
    config_path = Path(args.config).expanduser().resolve()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['__config_dir'] = str(config_path.parent)
        
    if args.subset:
        config['subset'] = args.subset
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    
    train_model(config, device)
