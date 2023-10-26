from torchvision.models.vgg import *
import torch.nn as nn
import torch



class VGG(nn.Module):
    # Available versions of the VGG model with associated architectures and weights
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
        """
        Initialize a VGG model.

        Args:
            version (str): Version of the model to use. Available: [ 11, 13, 16, 19 ]
            num_classes (int): Number of output classes.
            hidden_features (int, optional): Number of hidden features. Default: 4096
            dropout (float, optional): Dropout rate. Default: 0.0
            pretrained (bool, optional): Whether to use pre-trained weights. Default: False
            freeze (bool, optional): Whether to freeze model parameters. Default: False

        Raises:
            ValueError: If an unsupported version is specified.
        """
        super().__init__()

        # Store the parameters
        self.version = version
        self.num_classes = num_classes
        self.dropout = dropout
        self.pretrained = pretrained
        self.freeze = freeze

        # Check if version is available
        if version not in self.versions:
            raise ValueError(f"Versions available: {list(self.versions.keys())}")

        # Get the features layer
        model, weight = self.versions.get(version)
        self.features: nn.Module = model(
            weights = weight if pretrained else None,
            dropout = dropout,
        )

        # Freeze the features layer
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Replace the fully-connected layer
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
