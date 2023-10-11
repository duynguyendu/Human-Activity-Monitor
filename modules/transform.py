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

    transform = DataTransformation(image_size)

    transform(image)
    # or
    transform.DEFAULT(image)
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
    def AUGMENT_LV1(self):
        """ Survivor 🌟 """
        return T.Compose([
            T.RandomResizedCrop(self.image_size, antialias=True),
            T.RandomHorizontalFlip(p=0.1),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            ),
        ])


    @property
    def AUGMENT_LV2(self):
        """ Trooper 🔫 """
        return T.Compose([
            T.RandomResizedCrop(self.image_size, antialias=True),
            T.RandomHorizontalFlip(p=0.2),
            T.RandomRotation(30),
            T.GaussianBlur(3),
            T.RandomPerspective(0.2, p=0.2),
            T.RandomAutocontrast(p=0.2),
            T.RandomAdjustSharpness(2, p=0.2),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.075),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            ),
        ])


    @property
    def AUGMENT_LV3(self):
        """ Veterant 🚀 """
        return T.Compose([
            T.RandomResizedCrop(self.image_size, antialias=True),
            T.RandomHorizontalFlip(p=0.35),
            T.RandomRotation(60),
            T.GaussianBlur(5),
            T.RandomPerspective(0.35, p=0.35),
            T.RandomAutocontrast(p=0.35),
            T.RandomAdjustSharpness(4, p=0.35),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.RandomEqualize(p=0.35),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            ),
        ])


    @property
    def AUGMENT_LV4(self):
        """ Hellraiser 🗿 """
        return T.Compose([
            T.RandomResizedCrop(self.image_size, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(90),
            T.GaussianBlur(7),
            T.RandomPerspective(0.5, p=0.5),
            T.RandomAutocontrast(p=0.5),
            T.RandomAdjustSharpness(6, p=0.5),
            T.RandomAffine(degrees=30, translate=(0.1, 0.1)),
            T.RandomEqualize(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            ),
        ])


    @property
    def AUGMENT_LV5(self):
        """ Doom Slayer 💀 """
        return T.Compose([
            T.RandomResizedCrop(self.image_size, antialias=True),
            T.RandomHorizontalFlip(p=0.69),
            T.RandomRotation(96),
            T.GaussianBlur(9.6),
            T.RandomPerspective(0.69, p=0.69),
            T.RandomAutocontrast(p=0.69),
            T.RandomAdjustSharpness(9.6, p=0.69),
            T.RandomAffine(degrees=69, translate=(0.6, 0.9)),
            T.RandomEqualize(p=0.69),
            T.ColorJitter(brightness=0.69, contrast=0.69, saturation=0.69, hue=0.5),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            ),
        ])
