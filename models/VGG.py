import torch
import torch.nn as nn
from torchvision.models.vgg import (
    vgg11_bn, VGG11_BN_Weights, 
    vgg13_bn, VGG13_BN_Weights,
    vgg16_bn, VGG16_BN_Weights, 
    vgg19_bn, VGG19_BN_Weights,
)



class VGGModule(nn.Module):
    def __init__(
            self, features: nn.Module, num_classes: int,  hidden_features: int = 4096, dropout: int = 0.5, freeze: bool = False
        ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = features
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_features),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class VGG11(VGGModule):
    def __init__(
            self, num_classes: int, hidden_features: int = 4069, dropout: int = 0.5, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vgg11_bn(weights=VGG11_BN_Weights.DEFAULT if pretrained else None), 
            num_classes, hidden_features, dropout, 
            freeze if pretrained else False
        )


class VGG13(VGGModule):
    def __init__(
            self, num_classes: int, hidden_features: int = 4069, dropout: int = 0.5, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vgg13_bn(weights=VGG13_BN_Weights.DEFAULT if pretrained else None), 
            num_classes, hidden_features, dropout, 
            freeze if pretrained else False
        )


class VGG16(VGGModule):
    def __init__(
            self, num_classes: int, hidden_features: int = 4069, dropout: int = 0.5, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vgg16_bn(weights=VGG16_BN_Weights.DEFAULT if pretrained else None), 
            num_classes, hidden_features, dropout, 
            freeze if pretrained else False
        )


class VGG19(VGGModule):
    def __init__(
            self, num_classes: int, hidden_features: int = 4069, dropout: int = 0.5, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vgg19_bn(weights=VGG19_BN_Weights.DEFAULT if pretrained else None), 
            num_classes, hidden_features, dropout, 
            freeze if pretrained else False
        )
