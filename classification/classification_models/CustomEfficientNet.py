import torch
from torch import nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from classification_models.ClassifierBase import ClassifierBase


class CustomEfficientNet(nn.Module, ClassifierBase):
    def __init__(self,
        num_classes: int,
        start_from: str = None,
        pretrained: str = True,
        type_efficientnet: str = 'efficientnet-b0'):

        nn.Module.__init__(self)
        ClassifierBase.__init__(self, num_classes, start_from, pretrained)

        if self.pretrained:
            self.model = EfficientNet.from_pretrained(type_efficientnet)
        else:
            self.model = EfficientNet.from_name(type_efficientnet)

        # replace the prediction layer
        in_features = self.model._fc.in_features

        # defining dense top layers after the convolutional layers
        self.model._fc = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

        if self.start_from is not None:
            self.unfreeze_layers()


    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model.forward(tensor)


    @property
    def params_to_update(self):
        print('Parameters to update:')
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print(f"\t {name}")
                print(param.shape)

        return params_to_update


    def unfreeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if self.start_from is not None:
            model_children = [child_name for child_name, _child
                in self.model.named_children()]
            idx_first_to_train = model_children.index(self.start_from)
            children_to_train = model_children[idx_first_to_train:]

            # Unfreeze the parameters for certain children:
            for name, child in self.model.named_children():
                if name in children_to_train:
                    for parameter in child.parameters():
                        parameter.requires_grad = True


    def get_transform(self) -> transforms.Compose:
        transformation = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return transformation
