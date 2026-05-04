from enum import Enum


class ClassifierType(Enum):
    RESNET101 = 'Resnet101'
    RESNET50 = 'Resnet50'
    RESNET18 = 'Resnet18'
    EFFICIENTNET = 'EfficientNet'
    RESNET152 = 'Resnet152'