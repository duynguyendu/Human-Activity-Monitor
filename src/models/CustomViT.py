from torchvision.models.vgg import make_layers
from src.modules.model import LitModel
from .Vision_Transformer import *
import torch.nn as nn
import torch


__all__ = [
    "ViT_v1",
    "ViT_v2",
]


class ViT_v1(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.vit = ViT_B_32(num_classes, dropout, attention_dropout, pretrained, freeze)
        if freeze:
            for param in self.vit.model.parameters():
                param.requires_grad = False
        self.vit.model.heads = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


class ViT_v2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.vit = LitModel(
            ViT_B_32(num_classes, dropout, attention_dropout, pretrained, freeze),
            checkpoint="logs/new/ViT_B_32_ft_UTD/checkpoints/epoch=15-step=2992.ckpt",
        ).model
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.vit.model.conv_proj = make_layers(
            cfg=[48, "M", 96, "M", 192, "M", 384, "M", 768, "M"], batch_norm=True
        )
        self.vit.model.heads = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
