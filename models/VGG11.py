from torch import nn
from torchvision.models import vgg11



class VGG11(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vgg = vgg11(progress=True)
        self.fc = nn.Linear(3, 11)
        self.sofmax = nn.Softmax()

    def forward(self, x):
        out = self.vgg(x)
        out = self.fc(out)
        out = self.sofmax(out)
        return out


