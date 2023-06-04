import os
import torch
from models.wresnet import WideResNet
from utils.config import backdoor_PT_model_path


def select_model(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None,
                 trigger_pattern=None,
                 sel_model=None):
    if dataset.upper() in ['CIFAR10', 'GTSRB', 'PUBFIG']:
        if dataset.upper() == 'GTSRB':
            n_classes = 43
        else:
            n_classes = 10
        if model_name == 'WRN-16-1':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name == 'WRN-16-2':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0.0)
        elif model_name == 'WRN-40-1':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name == 'WRN-40-2':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0.0)
        elif model_name == 'VGG16':
            assert dataset.upper() == 'GTSRB', 'model is customized for GTSRB.'
            from models.vgg import VGG16
            model = VGG16(num_classes=n_classes)
        elif model_name == 'cifar_resnet20':
            from models.shufl_resnet import resnet20
            model = resnet20(n_classes)
        elif model_name == 'cifar_resnet18':
            from models.shufl_resnet import resnet18
            model = resnet18(n_classes)
        elif model_name == 'torch_resnet18':
            from models.torch_resnet import resnet18
            model = resnet18(weights='DEFAULT')
        elif model_name == 'torch_resnet34':
            from models.torch_resnet import resnet34
            model = resnet34(weights='DEFAULT')
        else:
            raise NotImplementedError(f"model: {model_name}")

        if pretrained:
            if trigger_pattern is not None:
                model_path = os.path.join(backdoor_PT_model_path, 
                dataset.lower(), model_name, trigger_pattern, pretrained_models_path,
                    f"{sel_model}.pth"
                )
            else:
                model_path = os.path.join(pretrained_models_path)
            print('Loading Model from {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                print(f' Load model entry.')
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
    else:
        raise NotImplementedError

    return model
