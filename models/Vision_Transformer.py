from typing import Any

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import (
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_l_32, ViT_L_32_Weights,
    vit_h_14, ViT_H_14_Weights,
)


__all__ = [
    "ViT_B_16", 
    "ViT_B_32", 
    "ViT_L_16", 
    "ViT_L_32", 
    "ViT_H_14"
]



class ViTModule(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int, freeze: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = features
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.heads = nn.Sequential(
            nn.Linear(768, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class ViT_B_16(ViTModule):
    def __init__(
            self, num_classes: int, dropout: float = 0.0, attention_dropout: float = 0.0, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vit_b_16(
                weights = ViT_B_16_Weights.DEFAULT if pretrained else None, 
                dropout = dropout, attention_dropout = attention_dropout
            ), 
            num_classes, freeze if pretrained else False
        )


class ViT_B_32(ViTModule):
    def __init__(
            self, num_classes: int, dropout: float = 0.0, attention_dropout: float = 0.0, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vit_b_32(
                weights = ViT_B_32_Weights.DEFAULT if pretrained else None, 
                dropout = dropout, attention_dropout = attention_dropout
            ), 
            num_classes, freeze if pretrained else False
        )


class ViT_L_16(ViTModule):
    def __init__(
            self, num_classes: int, dropout: float = 0.0, attention_dropout: float = 0.0, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vit_l_16(
                weights = ViT_L_16_Weights.DEFAULT if pretrained else None, 
                dropout = dropout, attention_dropout = attention_dropout
            ), 
            num_classes, freeze if pretrained else False
        )


class ViT_L_32(ViTModule):
    def __init__(
            self, num_classes: int, dropout: float = 0.0, attention_dropout: float = 0.0, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vit_l_32(
                weights = ViT_L_32_Weights.DEFAULT if pretrained else None, 
                dropout = dropout, attention_dropout = attention_dropout
            ), 
            num_classes, freeze if pretrained else False
        )


class ViT_H_14(ViTModule):
    def __init__(
            self, num_classes: int, dropout: float = 0.0, attention_dropout: float = 0.0, pretrained: bool = False, freeze: bool = False
        ) -> None:
        super().__init__(
            vit_h_14(
                weights = ViT_H_14_Weights.DEFAULT if pretrained else None, 
                dropout = dropout, attention_dropout = attention_dropout
            ), 
            num_classes, freeze if pretrained else False
        )
