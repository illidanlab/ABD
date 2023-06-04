import torch
import torchvision.transforms as transforms
from .backdoor import BackdooredDataset
from utils.config import DATA_PATHS


def get_test_loader(args, shuffle=False, poi_shuffle=False):
    dataset_path = DATA_PATHS[args.dataset]
    if args.dataset.upper() in ['CIFAR10', 'GTSRB']:

        if args.dataset.lower() == 'cifar10':
            from torchvision.datasets import CIFAR10
            num_classes = 10
            DATA_CLASS = CIFAR10

            IMG_MEAN = (0.4914, 0.4822, 0.4465)
            IMG_STD = (0.2023, 0.1994, 0.2010)
        elif args.dataset.lower() == 'gtsrb':
            from .gtsrb import GTSRB
            num_classes = 43
            DATA_CLASS = GTSRB

            IMG_MEAN = (0.3403, 0.3121, 0.3214)
            IMG_STD = (0.2724, 0.2608, 0.2669)
        else:
            raise NotImplementedError(f"args.dataset: {args.dataset}")
        
        if args.norm_inp:
            # mean = (0.4914, 0.4822, 0.4465)
            # std = (0.2023, 0.1994, 0.2010)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_STD)])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor()])
        # clean set
        test_dataset = DATA_CLASS(
            dataset_path, train=False, download=True, transform=transform_test)

        # poisoned set
        test_poisoned_set = BackdooredDataset(data_root_path=dataset_path,
                                            dataset_name=args.dataset,
                                            split='test', triggered_ratio=0,
                                            trigger_pattern=args.trigger_pattern,
                                            target=args.poi_target, transform=transform_test)

    elif args.dataset == 'pubfig':
        from .pubfig import PubFig, BackDoorPubFig
        num_classes = 83
        # based on ImageNet.
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        test_transform = [transforms.ToTensor()]
        if args.norm_inp:
            test_transform.append(transforms.Normalize(IMG_MEAN, IMG_STD))
        test_transform = transforms.Compose(test_transform)

        # poisoned set
        test_poisoned_set = BackDoorPubFig(
            dataset_path, train=False, transform=test_transform, ratio_holdout=0.,
            triggered_ratio=1., trigger_pattern=args.trigger_pattern,  attack_target=args.poi_target,
        )

        # clena set
        test_dataset = PubFig(dataset_path, train=False,
                              transform=test_transform)

    else:
        raise NotImplementedError

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size,
        shuffle=shuffle, drop_last=False, num_workers=args.workers,
        pin_memory=False)

    test_poi_loader = torch.utils.data.DataLoader(
        dataset=test_poisoned_set, batch_size=args.batch_size,
        shuffle=poi_shuffle, drop_last=False, num_workers=args.workers,
        pin_memory=False)
    return test_loader, test_poi_loader
