from typing import List
from pathlib import Path
import torch
from torchvision import transforms
from abc import ABC, abstractmethod

class ClassifierBase(ABC):
    def __init__(self,
        num_classes: int,
        start_from: str = None,
        pretrained: str = True,
        custom_weights_path: Path = None):

        self.num_classes = num_classes
        self.start_from = start_from
        self.pretrained = pretrained
        self.custom_state_dict = None
        if custom_weights_path is not None:
            self.custom_state_dict = self._adjust_state_dict_layer_names(custom_weights_path)

    @abstractmethod
    def params_to_update(self) -> List[str]:
        pass

    @abstractmethod
    def get_transform(self) -> transforms.Compose:
        pass

    @staticmethod
    def _adjust_state_dict_layer_names(custom_weights_path: Path):
        state_dict = torch.load(custom_weights_path)
        trained_model_dict = {
            k.replace('model.', ''): v for k,v in  state_dict['model_state_dict'].items()
        }
        return trained_model_dict
