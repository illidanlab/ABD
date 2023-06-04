import torch
from torch import nn

cfg = {'small_VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],}
drop_rate = [0.3,0.4,0.4]

class VGG16(nn.Module):
    """VGG for GTSRB. Based on https://github.com/YiZeng623/I-BAU/blob/main/clean_solution_batch_opt_hyperdimentional_gtsrb.ipynb"""
    def __init__(self, num_classes=43):
        super(VGG16, self).__init__()
        self.features = self._make_layers(cfg['small_VGG16'])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.features(x)
        features = out
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, features

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        key = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(drop_rate[key])]
                key += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ELU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
