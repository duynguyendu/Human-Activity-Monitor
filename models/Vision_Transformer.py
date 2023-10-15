from torchvision.models.vision_transformer import *
import torch.nn as nn
import torch



class ViT(nn.Module):
    # Available versions of the ViT model with associated architectures and weights
    versions = {
        "B_16": ( vit_b_16, ViT_B_16_Weights.DEFAULT ),
        "B_32": ( vit_b_32, ViT_B_32_Weights.DEFAULT ),
        "L_16": ( vit_l_16, ViT_L_16_Weights.DEFAULT ),
        "L_32": ( vit_l_32, ViT_L_32_Weights.DEFAULT ),
        "H_14": ( vit_h_14, ViT_H_14_Weights.DEFAULT ),
    }

    def __init__(
            self,
            version: str,
            num_classes: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            pretrained: bool = False,
            freeze: bool = False
        ) -> None:
        """
        Initialize a Vision Transformer (ViT) model.

        Args:
            version (str): Version of the ViT model to use. Available: [ B_16, B_32, L_16, L_32, H_14 ]
            num_classes (int): Number of output classes.
            dropout (float, optional): Dropout rate. Default is 0.0.
            attention_dropout (float, optional): Attention dropout rate. Default is 0.0.
            pretrained (bool, optional): Whether to use pre-trained weights. Default is False.
            freeze (bool, optional): Whether to freeze model parameters. Default is False.

        Raises:
            ValueError: If an unsupported version is specified.

        """
        super().__init__()
        self.num_classes = num_classes

        if version not in self.versions:
            raise ValueError(f"Versions available: {list(self.versions.keys())}")

        model, weight = self.versions.get(version)
        self.features: nn.Module = model(weights=weight if pretrained else None, dropout=dropout, attention_dropout=attention_dropout)

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.features.heads = nn.Linear(768, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
