from torchvision.models.vgg import *
import torch.nn as nn
import torch



class VGG(nn.Module):
    versions = {
        "11": ( vgg11_bn, VGG11_BN_Weights.DEFAULT ),
        "13": ( vgg13_bn, VGG13_BN_Weights.DEFAULT ),
        "16": ( vgg16_bn, VGG16_BN_Weights.DEFAULT ),
        "19": ( vgg19_bn, VGG19_BN_Weights.DEFAULT )
    }

    def __init__(
            self,
            version: str,
            num_classes: int,
            hidden_features: int = 4096,
            dropout: int = 0.0,
            pretrained: bool = False,
            freeze: bool = False
        ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if version not in self.versions:
            raise ValueError(f"Versions available: {list(self.versions.keys())}")

        model, weight = self.versions.get(version)
        self.features: nn.Module = model(weights=weight if pretrained else None, dropout=dropout)

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.features.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_features),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
