import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Generator(nn.Module):

    def __init__(self, z_dim, target_size=32):
        super(Generator, self).__init__()

        layers = [
            nn.Linear(z_dim, 128 * 8**2),
            View((-1, 128, 8, 8)),
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),  # 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if target_size == 32:
            layers += [
                nn.Conv2d(64, 3, 3, stride=1, padding=1),

                nn.BatchNorm2d(3, affine=True)
            ]
        elif target_size == 224:
            layers += [
                nn.Upsample(scale_factor=2),  # 64
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Upsample(scale_factor=2),  # 128
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Upsample(size=224),  # 224
                nn.Conv2d(16, 8, 3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(8, 3, 3, stride=1, padding=1),

                nn.BatchNorm2d(3, affine=True)
            ]
        elif target_size == 256:
            layers += [
                nn.Upsample(scale_factor=2),  # 64
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Upsample(scale_factor=2),  # 128
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Upsample(scale_factor=2),  # 256
                nn.Conv2d(16, 8, 3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(8, 3, 3, stride=1, padding=1),

                nn.BatchNorm2d(3, affine=True)
            ]
        else:
            raise NotImplementedError()

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)
