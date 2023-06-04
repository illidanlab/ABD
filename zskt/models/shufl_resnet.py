import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
try:
    from advertorch.utils import NormalizeByChannelMeanStd
except:
    pass


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNets(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, contain_normalize=False):
        super(ResNets, self).__init__()
        self.in_planes = 16

        # print('The normalize layer is contained in the network')
        # self.normalize = NormalizeByChannelMeanStd(
        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.contain_normalize = contain_normalize
        if contain_normalize:
            # raise NotImplementedError()
            print('The normalize layer is contained in the network')
            self.normalize = NormalizeByChannelMeanStd(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.contain_normalize:
            x = self.normalize(x)

        # x = self.normalize(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        act1 = out
        out = self.layer2(out)
        act2 = out
        out = self.layer3(out)
        act3 = out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.fc(out), act1, act2, act3

def resnet20(num_class, contain_normalize=False):
    return ResNets(BasicBlock, [3, 3, 3], num_classes=num_class, contain_normalize=contain_normalize)


def resnet18(num_class, contain_normalize=False):
    return ResNets(BasicBlock, [2, 2, 2, 2], num_classes=num_class, contain_normalize=contain_normalize)
