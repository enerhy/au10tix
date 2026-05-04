from typing import List
import torch
from torch import nn
import torchvision
from torchvision import transforms
from  torchvision.models.resnet import ResNet, Bottleneck
from pathlib import Path
from abc import ABC, abstractmethod
from classification_models.ClassifierBase import ClassifierBase


class CustomResnetBase(nn.Module, ClassifierBase):
    '''
    Resnet Base class for all resnets.
    Args:
        num_classes (int): num classes in the final layer
        start_from (init): starting block for fine-tunning, e.g. layer4
        pretrained (boolen): True or False
        custom_weights_path: the file path 'model.pt' containing the state 
        dict of the model
    '''
    def __init__(self,
        num_classes: int,
        start_from: str = None,
        pretrained: str = True,
        custom_weights_path: Path =  None):

        nn.Module.__init__(self)
        ClassifierBase.__init__(self, num_classes, start_from, pretrained, custom_weights_path)

        if pretrained and self.custom_state_dict is not None: #True & not None
            self.model = self._load_model_from_custom_pretrained()
        else: # True & None; False
            self.model = self._create_model()

        # replace the prediction layer if there is a mismatch 
        # between the loaded out_features for the model and 
        # the input num_classes
        if self.model.fc.out_features != self.num_classes:
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)

        if self.start_from is not None:
            self.unfreeze_layers()


    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def _load_model_from_custom_pretrained(self):
        pass

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model.forward(tensor)


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


    def get_transform(self) -> transforms.Compose:
        # Resize to 256x256, then center-crop to 224x224 (to match the resnet image size)
        # TODO: to use resnet transform but present in which version of pytorch?!
        transformation = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transformation


    def _get_custom_pretrained_model_with_layers(self, layers: List[int]):
        #TODO: check custom state dict matches resnet model
        # if weights match model
        #else warn
        num_output_classes = len(self.custom_state_dict['fc.weight'])
        model = ResNet(Bottleneck, layers, num_output_classes)
        model.load_state_dict(self.custom_state_dict)
        return model


class CustomResnet101(CustomResnetBase):

    def _create_model(self):
        return torchvision.models.resnet101(pretrained = self.pretrained)

    def _load_model_from_custom_pretrained(self):
        return self._get_custom_pretrained_model_with_layers([3, 4, 23, 3])


class CustomResnet18(CustomResnetBase):

    def _create_model(self):
        return torchvision.models.resnet18(pretrained = self.pretrained)

    def _load_model_from_custom_pretrained(self):
        return self._get_custom_pretrained_model_with_layers([2, 2, 2, 2])


class CustomResnet50(CustomResnetBase):

    def _create_model(self):
        return torchvision.models.resnet50(pretrained = self.pretrained)

    def _load_model_from_custom_pretrained(self):
        return self._get_custom_pretrained_model_with_layers([3, 4, 6, 3])


class CustomResnet152(CustomResnetBase):

    def _create_model(self):
        return torchvision.models.resnet152(pretrained = self.pretrained)

    def _load_model_from_custom_pretrained(self):
        return self._get_custom_pretrained_model_with_layers([3, 8, 36, 3])