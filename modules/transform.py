from typing import Tuple
from PIL import Image

import torchvision.transforms as T
from torch import Tensor




class DataTransformation:
    """
    Contain some available transfrom for image
    
    Usage
    -----
    ```
    image = ...
    image_size = (224, 224)

    transform = DataAugmentation(image_size)

    transform(image)
    # or
    transform.DEFAULT(image)
    # or
    transform.CUSTOM(image)
    ```
    """
    def __init__(self, image_size: Tuple=(224, 224)) -> None:
        self.image_size = image_size


    def __call__(self, image: Image) -> Tensor:
        """
        Apply DEFAULT transform
        """
        return self.DEFAULT(image)


    @property
    def DEFAULT(self):
        return T.Compose([
            T.Resize(self.image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            ),
        ])


    @property
    def TOPIL(self):
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            ),
        ])


    @property
    def AUGMENTATION(self):
        return T.Compose([
            T.RandomResizedCrop(self.image_size, antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.GaussianBlur(3),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            ),
        ])
