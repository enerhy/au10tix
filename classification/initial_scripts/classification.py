'''
TODO: in save best model as .pt save also epoch number and loss if possible
'''
# general
from __future__ import print_function, division
import os
import copy

from confusion_matrix import ConfusionMatrixWithMetrics

# libraries
import yaml
import time
import click
import numpy as np
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
import mlflow
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from pathlib import Path
import pickle

# pytorch
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import transforms

# custom
from CustomLogger import CustomLogger
from data_importers import CustomCSVReader

# custom libraries
from imagga_models.classification.ClassifierBase import ClassifierBase
from imagga_models.classification.CustomResnet import (
    CustomResnet18, CustomResnet50, CustomResnet101) #, CustomResnet152)
from imagga_models.classification.CustomEfficientNet import CustomEfficientNet
from imagga_models.classification.ClassifierType import ClassifierType

cudnn.benchmark = True

# mlflow experiments
current_filepath = Path(os.path.abspath(__file__)).parent

env_filepath = current_filepath / '..' / '.env'
# env_filepath = current_filepath / '.env'
load_dotenv(env_filepath)

SEED = 42
np.random.seed(SEED)

INITIAL_LOSS = 1e10
NUM_WORKERS = 194
ITERS_PER_VAL_METRICS_LOG = 1
#SAVE_MODELS_WITH_CONFUSION_MATRIX_METRICS_REPORT = False


def get_saved_model(
    model: ClassifierBase,
    optimizer: torch.optim.Optimizer,
    saved_model_path: str,
    logger: CustomLogger) -> ClassifierBase:
    start_epoch = 1
    start_loss = INITIAL_LOSS

    # load saved model if available
    if (saved_model_path is not None and
            Path(saved_model_path).exists() and
            Path(saved_model_path).is_file()):

        checkpoint = torch.load(saved_model_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        start_loss = checkpoint['loss']

        logger.info(
            'Loaded model to continue training' +
            'from file="{saved_model_path}"' +
            'from epoch={start_epoch}')


    return model, optimizer, start_epoch, start_loss


def inference(model, criterion, optimizer, phase, inputs, labels):
    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        # backward pass + optimize only if in training phase
        if phase == 'train':
            loss.backward()
            optimizer.step()

    return (preds, loss)


def inference_and_calculate_loss(
    model, dataloaders, criterion, optimizer,
    device, class_labels, phase):

    running_loss = 0
    running_corrects = 0

    num_classes = len(class_labels)
    confusion_matrix_array = np.zeros((num_classes, num_classes))

    for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds, loss = inference(
            model, criterion, optimizer, phase, inputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

        true_labels = labels.detach().cpu().numpy()
        predicted_labels = preds.detach().cpu().numpy()

        if phase == 'val':
            for i, j in zip(true_labels, predicted_labels):
                confusion_matrix_array[i, j] += 1

    confusion_matrix = None
    if phase == 'val':
        confusion_matrix = ConfusionMatrixWithMetrics(
            class_labels, confusion_matrix_array)

    return (running_loss, running_corrects, confusion_matrix)


def train_per_epoch(
    model, dataloaders, dataset_sizes,
    class_names, criterion, optimizer,
    scheduler, device, output_dir, logger,
    end_epoch, epoch):

    saved_model_acc = 0.0

    current_lr = optimizer.param_groups[0]['lr']
    mlflow.log_metric(key='lr_value', value=current_lr, step=epoch)

    logger.info(f'Epoch {epoch}/{end_epoch}')
    logger.info('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        (running_loss, running_corrects, confusion_matrix) = \
            inference_and_calculate_loss(
                model, dataloaders, criterion, optimizer,
                device, class_names, phase)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

        if phase == 'train' and isinstance(scheduler,  StepLR):
            scheduler.step()
        
        elif phase == 'val' and isinstance(scheduler,  ReduceLROnPlateau):
            scheduler.step(epoch_loss)

        logger.info(
            f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')

        mlflow.log_metric(key=f'{phase}_loss', value=epoch_loss, step=epoch)
        mlflow.log_metric(
            key=f'{phase}_accuracy', value=epoch_accuracy, step=epoch)

        # save a checkpoint on validation per num epochs
        if phase == 'val' and \
            (epoch % ITERS_PER_VAL_METRICS_LOG == 0 or epoch == end_epoch):
            saved_model_acc = epoch_accuracy

            confusion_matrix.mlflow_log_metrics(epoch)

            #if SAVE_MODELS_WITH_CONFUSION_MATRIX_METRICS_REPORT:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss},
                str(Path(output_dir) / f'model{epoch}.pt'))

            confusion_matrix.export_confusion_matrix_with_reports(
                str(epoch), output_dir)

    return saved_model_acc


def log_confusion_matrix_reports(
        confusion_matrix: ConfusionMatrixWithMetrics,
        best_epoch: int,
        output_dir: str):
    cm_image_filepath = \
        confusion_matrix.export_confusion_matrix_with_metrics_table(
        f'cm_best_epoch{best_epoch}_with_scores.png', output_dir)
    mlflow.log_artifact(cm_image_filepath)

    cm_csv_path, report_csv_path, report_html_path = \
        confusion_matrix.export_confusion_matrix_with_reports(
        str(best_epoch), output_dir)
    mlflow.log_artifact(cm_csv_path)
    mlflow.log_artifact(report_csv_path)
    mlflow.log_artifact(report_html_path)


def log_best_and_last_model(
        model, optimizer, best_epoch, end_epoch, output_dir):

    # Save model (best and last)

    # 1. the whole torch model (in mlflow -> 'model.pth' with artefacts)
    torch.save(model, str(Path(output_dir) / 'best_model.pth'))
    mlflow.pytorch.log_model(model, 'best_model')

    # save as .pt as well:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        },
        str(Path(output_dir) / 'best_model.pt'))

    # 2. the weights to continue training
    mlflow.log_artifact(
        str(Path(output_dir) / f'model{best_epoch}.pt'), 'best_model')

    # save last model as weights
    mlflow.log_artifact(
        str(Path(output_dir) / f'model{end_epoch}.pt'), 'last_model')


def train_model(
    model, dataloaders, dataset_sizes, class_names,
    criterion, optimizer, scheduler,
    start_epoch, device,
    num_epochs, output_dir, logger):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = start_epoch
    end_epoch = start_epoch + num_epochs - 1

    for epoch in range(start_epoch, end_epoch+1):
        saved_model_acc = train_per_epoch(
            model, dataloaders, dataset_sizes, class_names,
            criterion, optimizer, scheduler,
            device, output_dir, logger, end_epoch, epoch)

        if(saved_model_acc > best_acc):
            best_acc = saved_model_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    mins = time_elapsed // 60
    secs = time_elapsed % 60
    logger.info(f'Training completed in {mins:.0f}m {secs:.0f}s')
    logger.info(f'Best val accuracy: {best_acc:4f}')

    # load best model
    model.load_state_dict(best_model_wts)

    # calculate metrics with the best model
    _, _, confusion_matrix = inference_and_calculate_loss(
        model, dataloaders, criterion, optimizer,
        device, class_names, phase='val')

    log_confusion_matrix_reports(confusion_matrix, best_epoch, output_dir)

    log_best_and_last_model(
        model, optimizer, best_epoch, end_epoch, output_dir)


TRANFORM_DICT = {
    'horizontal_flip': lambda kwargs:
        transforms.RandomHorizontalFlip(**kwargs),
    'vertical_flip': lambda kwargs:
        transforms.RandomVerticalFlip(**kwargs),
    'color_jiter': lambda kwargs:
        transforms.ColorJitter(**kwargs),
    'random_affine': lambda kwargs:
        transforms.RandomAffine(**kwargs),
    'random_rotation': lambda kwargs:
        transforms.RandomRotation(**kwargs)
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


def load_dataset_from_csv(
    csv_file: str,
    model: ClassifierBase,
    batch_size: int,
    augment_dict: dict):

    train_dataset, val_dataset = None, None
    train_transform = model.get_transform()
    val_transform = model.get_transform()

    if augment_dict['train'] is True:
        train_transform = augment_transform(augment_dict, train_transform)
    if augment_dict['valid'] is True:
        val_transform = augment_transform(augment_dict, val_transform)

    class_names = []
    class_to_idx = {}

    # Initialize datasets
    train_dataset = CustomCSVReader(csv_file=csv_file, split='train',
                                    relative_paths=False, root_dir=None,
                                    transform=train_transform)

    val_dataset = CustomCSVReader(csv_file=csv_file, split='val',
                                  relative_paths=False, root_dir=None,
                                  transform=val_transform)
    class_names = train_dataset.class_names
    class_to_idx = train_dataset.class_to_idx

    image_datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                    batch_size=batch_size,
                    collate_fn=custom_collate_fn,
                    shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=True)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    mlflow.log_param('class_names', class_names)
    mlflow.log_param('class_to_idx', class_to_idx)
    mlflow.log_param('train_size', dataset_sizes['train'])
    mlflow.log_param('val_size', dataset_sizes['val'])

    return dataloaders, image_datasets, dataset_sizes, class_names, class_to_idx


def load_dataset_from_dir(
    data_dirpath: str,
    model: ClassifierBase,
    train_percentage: float,
    batch_size: int,
    augment_dict: dict):

    train_dataset, val_dataset = None, None
    train_dir = Path(data_dirpath) / 'train'
    val_dir = Path(data_dirpath) / 'val'

    train_transform = model.get_transform()
    val_transform = model.get_transform()

    if augment_dict['train'] is True:
        train_transform = augment_transform(augment_dict, train_transform)
    if augment_dict['valid'] is True:
        val_transform = augment_transform(augment_dict, val_transform)  

    class_names = []
    class_to_idx = {}

    if  train_dir.exists() and \
        val_dir.exists():

        train_dataset = torchvision.datasets.ImageFolder(
            root=train_dir, transform = train_transform)
        val_dataset = torchvision.datasets.ImageFolder(
            root=val_dir, transform = val_transform)
        
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx
    else:
        # No augmentation possible
        full_dataset = torchvision.datasets.ImageFolder(
            root=data_dirpath, transform = val_transform)

        train_size = int(train_percentage * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED))
        
        class_names = full_dataset.classes
        class_to_idx = full_dataset.class_to_idx

    image_datasets = {'train': train_dataset, 'val': val_dataset}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  collate_fn=custom_collate_fn,
                                                  shuffle=True,
                                                  num_workers=NUM_WORKERS)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = sorted(class_names)

    # logger.info(f'class names: {class_names}')
    # logger.info(f'class to idx: {class_to_idx}')
    # logger.info(f'train size: {dataset_sizes["train"]}')
    # logger.info(f'val size {dataset_sizes["val"]}')

    mlflow.log_param('class_names', class_names)
    mlflow.log_param('class_to_idx', class_to_idx)
    mlflow.log_param('train_size', dataset_sizes['train'])
    mlflow.log_param('val_size', dataset_sizes['val'])

    return dataloaders, image_datasets, dataset_sizes, class_names, class_to_idx


def custom_collate_fn(batch):
    # Filter out samples where image or label is None
    batch = [b for b in batch if b[0] is not None and b[1] is not None]
    
    if len(batch) == 0:
        return None, None  # Handle the case where all images were invalid

    images, labels = zip(*batch)
    images = torch.stack(images)  # If images are already tensors
    labels = torch.tensor(labels)
    return images, labels

''' ------------- WORKING HERE ------------ '''
def get_loss(loss_type, data_input, train_dataset, class_to_idx, device):
    if loss_type == 'WeightedCrossEntropy':

        if os.path.isfile(data_input):
            category_counts = train_dataset.category_counts
            total_count = train_dataset.total_count


            # df = pd.read_csv(data_input, usecols=['category', 'split'])
            # df_train = df[df['split'] == 'train']
            # category_counts = df_train['category'].value_counts()
            # total_count = category_counts.sum()
            # category_counts = category_counts.to_dict()

        elif os.path.isdir(data_input):
            category_counts, total_count = get_train_class_counts(train_dataset)

        for category, count in category_counts.items():
            print(f'Category {category} has {count} samples')

        weights = {}
        class_names = class_to_idx.keys()
        for class_name, class_count in category_counts.items():
            weights[class_name] = \
                (1 / class_count) * (total_count / len(class_names))

        # use dataloader class_to_idx to define in the same the class weights
        weight_values = []
        for class_name, class_idx in class_to_idx.items():
            weight_values.insert(class_idx, weights[class_name])

        class_weights = torch.tensor(weight_values, 
            dtype = torch.float32).to(device)
        
        print('class counts: ', category_counts)
        print('CrossEntropyLoss weights: ', class_weights)
        
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    # else
    return torch.nn.CrossEntropyLoss()


def get_optimizer(
    params_to_update,
    optimizer_type: str,
    learning_rate: float,
    momentum: float,
    l2_penalty: float = 0.0
    ):
    optimizer = None
    # specify the optimizer
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(params_to_update, learning_rate, momentum)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(
            params_to_update, learning_rate, weight_decay=l2_penalty)

    return optimizer


def get_learning_rate_scheduler(optimizer, lr_schduler_type: str):
    
    if lr_schduler_type == 'reduce-on-plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='min',factor=0.2, patience=5,verbose=1)
    elif lr_schduler_type == 'stepwise':
        # decay LR by a factor of 0.1 every 7 epochs
        lr_scheduler = StepLR(
            optimizer, step_size=7, gamma=0.1)

    return lr_scheduler


def predict(model_filepath: str,
            image_filepath: str,
            data_dir: str,
            device: torch.device,
            config: dict,
            logger: CustomLogger):

    class_names = get_class_names(data_dir)
    model = get_model(config['model'], len(class_names), start_from=None)

    model = model.to(device)

    optimizer = get_optimizer(
        model.params_to_update,
        optimizer_type=config['optimizer']['type'],
        learning_rate=float(config['optimizer']['learning_rate']),
        momentum=float(config['optimizer']['momentum']),
        l2_penalty=config['l2'])

    #TODO: Check for error in the next function
    model, optimizer, _, _ = get_saved_model(
        model, optimizer, model_filepath, logger)
    
    model.eval()

    image = Image.open(image_filepath)

    data = model.get_transform()(image)

    input_batch = data.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

        softmax = nn.Softmax(dim=1)
        output = softmax(output)

        output = output.cpu().detach().numpy()[0]
        indices = output.argsort()

        prediction = []
        classes = []
        for i in indices:
            prediction.append(float(output[i]*100))
            classes.append(class_names[i])

        print(classes)
        print(prediction)


@click.command()
@click.option('--inputimage',
              required=True,
              help='The path to the input image to predict.')
@click.option('--inputdir',
              required=True,
              help='The path to the input dataset directory.')
@click.option('--outputdir',
              required=True,
              help='The path to the output directory.')
@click.option('--configfile',
              required=True,
              help='The path to the config file.')
@click.option('--gpu',
              default=0,
              type=int,
              required=False,
              help='The GPu number on which the model is trained.')
def predict_args(inputimage, inputdir, outputdir, configfile, gpu):
    '''
    Example call:
    python classification.py \
    --inputimage '/media/imagga/datasets/clients2/LorenNetwork/val/test.jpg' \
    --inputdir '/media/imagga/datasets/clients2/LorenNetwork/' \
    --outputdir './out_LorenNetwork' \
    --configfile 'classification-config.yaml' \
    --gpu 2
    '''
    # process input parameters
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    logger = CustomLogger(str(Path(outputdir) /'prediction.log'))

    if not os.path.exists(inputdir):
        logger.error(f'Not existing input dir: {inputdir}')
        return

    config = None
    with open(configfile, 'r', encoding='utf-8') as yamlfile:
        config = yaml.load(yamlfile, yaml.FullLoader)
        print('Read successful')
        print(config)
    if(config is None):
        logger.error(f'Not existing config file: {configfile}')
        return

    device = torch.device(
        f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    model_filepath = config['model_path']

    predict(model_filepath, inputimage, inputdir, device, config, logger)

# Q? block below
def get_class_names(data_input: str):

    if os.path.isfile(data_input):
        # Load only the 'class' column from the CSV
        df = pd.read_csv(data_input, usecols=['category'])
        class_names = df['category'].dropna().unique().tolist()

    elif os.path.isdir(data_input):

        dirpath = Path(data_input)

        train_dir = dirpath / 'train'
        val_dir = dirpath / 'val'

        if  train_dir.exists() and \
            val_dir.exists():
            dirpath = train_dir

        class_names = []
        for subdirpath in dirpath.iterdir():
            class_names.append(subdirpath.name)

    return class_names

#TODO: HERE IT STUCKS
def get_train_class_counts(train_dataset: torch.utils.data.IterableDataset):

    total_count = len(train_dataset)

    labels = [label for _, label in train_dataset]
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    class_counts = Counter(labels)
    class_counts_named = {idx_to_class[class_idx]: count for
                          class_idx, count in class_counts.items()}

    return class_counts_named, total_count


def get_model(
    model_config: dict, num_classes: int,
    start_from: str ) -> ClassifierBase:
    #TODO: Make this with alternative constructor
    start_from = model_config['train_start']
    pretrained = bool(model_config['pretrained'])
    
    if model_config['name'] == ClassifierType.EFFICIENTNET.value:
        type_efficientnet  = model_config['type_efficientnet']
        return CustomEfficientNet(
            num_classes = num_classes,
            start_from = start_from,
            pretrained = pretrained,
            type_efficientnet = type_efficientnet)
    elif model_config['name'] == ClassifierType.RESNET101.value:
        return CustomResnet101(
            num_classes = num_classes,
            start_from = start_from,
            pretrained = pretrained)
    elif model_config['name'] == ClassifierType.RESNET50.value:
        return CustomResnet50(
            num_classes = num_classes,
            start_from = start_from,
            pretrained = pretrained)
    elif model_config['name'] == ClassifierType.RESNET18.value:
        return CustomResnet18(
            num_classes = num_classes,
            start_from = start_from,
            pretrained = pretrained)
    elif model_config['name'] == ClassifierType.RESNET152.value:
        return CustomResnet152(
            num_classes = num_classes,
            start_from = start_from,
            pretrained = pretrained,
            custom_weights_path = Path(model_config['custom_weights_path']))


def main(config: dict, data_input, outputdir, gpu, logger):

    model_config = config['model']
    logger.info(f'Training data using {model_config}')

    device = torch.device(
        f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    class_names = get_class_names(data_input)
    model = get_model(
        model_config, len(class_names),
        start_from = model_config['train_start'])

    print('--- model loaded')
    if os.path.isdir(data_input):
        dataloaders, image_datasets, dataset_sizes, class_names, class_to_idx=\
            load_dataset_from_dir(
                data_input,
                model=model,
                train_percentage=float(config['train_percentage']),
                batch_size=int(config['batch_size']),
                augment_dict=config['augment'])
    else:
        dataloaders, image_datasets, dataset_sizes, class_names, class_to_idx=\
            load_dataset_from_csv(
                data_input,
                model=model,
                batch_size=int(config['batch_size']),
                augment_dict=config['augment'])
    
    print('--- data loaders created')

    # Log transformation
    tranform = model.get_transform()
    transform_file_path = os.path.join(outputdir, 'transform_object')
    with open(transform_file_path, 'wb') as transform_object:
        pickle.dump(tranform, transform_object)
    mlflow.log_artifact(transform_file_path, 'transform')
    if os.path.isfile(transform_file_path):
        os.remove(transform_file_path)
    model = model.to(device)

    criterion = get_loss(
        loss_type=config['loss']['type'], 
        data_input=data_input,
        train_dataset=image_datasets['train'],
        class_to_idx=class_to_idx,
        device=device)

    optimizer = get_optimizer(
        model.params_to_update,
        optimizer_type=config['optimizer']['type'],
        learning_rate=float(config['optimizer']['learning_rate']),
        momentum=float(config['optimizer']['momentum']),
        l2_penalty=config['l2'])

    lr_scheduler = get_learning_rate_scheduler(optimizer,
            config['optimizer']['lr_schduler_type'])

    model, optimizer, start_epoch, _start_loss = get_saved_model(
        model, optimizer, config['model_path'], logger)


    mlflow.log_text(repr(model), 'model.txt')

    print('--- ready for training')
    train_model(
        model,
        dataloaders,
        dataset_sizes,
        class_names,
        criterion,
        optimizer,
        lr_scheduler,
        start_epoch,
        device,
        int(config['epochs']),
        outputdir,
        logger)


def read_yaml_file(yaml_filepath: str, logger: CustomLogger):
    file_content = None
    with open(yaml_filepath, 'r', encoding='utf-8') as yamlfile:
        file_content = yaml.load(yamlfile, yaml.FullLoader)
        logger.info('Read successful')
        logger.info(file_content)
    return file_content


@click.command()
@click.option('--inputdir',
              default=None,
              required=False,
              help='The path to the input dataset directory.')
@click.option('--inputfile',
              default=None,
              required=False,
              help='The path to the file containing path, category, split for each image to be included.')
@click.option('--outputdir',
              required=True,
              help='The path to the output directory.')
@click.option('--configfile',
              required=True,
              help='The path to the config file.')
@click.option('--mlflow_experiment',
               help='The name of the MLFlow experiment.')
@click.option('--gpu',
              default=0,
              type=int,
              required=False,
              help='The GPU number on which to train.')
def main_args(inputdir, inputfile, outputdir, configfile, mlflow_experiment, gpu):
    '''
    Example call:
    python classification.py
    --configfile 'classification-config.yaml'
    --inputdir '/media/imagga/datasets/content-moderation2/inf_syms_dataset_v1'
    --inputfile '/home/aspasov/data/imagga-datasets/soe/adult/meta_data_export.csv'
    --outputdir './out_efficient_infamous'
    --mlflow_experiment 'CounteR-classification-infamous-landmarks'
    --gpu 2
    '''
    now = datetime.now()
    run_name = f'ID_{now.month}{now.day}{now.hour}{now.minute}{now.second}'

    outputdir = Path(outputdir) / run_name
    outputdir.mkdir(parents=True, exist_ok=True)

    logger = CustomLogger(outputdir / 'train.log')

    if inputfile is None and inputdir is None:
        logger.error('inputfile of inputdir must be parsed')
        return
    else:
        data_input = inputfile if not None else inputdir
    if not os.path.exists(data_input):
        logger.error(f'Not existing input: {data_input}')
        return

    config = read_yaml_file(configfile, logger)
    if config is None:
        logger.error(f'Not existing config file: {configfile}')
        return

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    if mlflow_experiment and mlflow_experiment != '':
        mlflow.set_experiment(mlflow_experiment)
    mlflow.autolog(log_input_examples=True)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(f'./{configfile}')
        mlflow.log_params(config)
        mlflow.log_param('data_input', data_input)
        mlflow.log_param('results_dir', outputdir)
        #TODO: add split_uid to the params!

        main(config, data_input, outputdir, gpu, logger)


if __name__ == '__main__':
    # trunk-ignore(pylint/E1120)
    main_args()
    # predict_args()
