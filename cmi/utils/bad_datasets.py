import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.io import imsave, imread
from torchvision.datasets import CIFAR10, CIFAR100
import torch
import torch.utils.data
from utils.gtsrb import GTSRB

from utils.config import BDBlocker_path


def _normalization(data):
    '''Normalize to [0,1].'''
    _min = np.min(data, axis=(1, 2, 3), keepdims=True)
    _max = np.max(data, axis=(1, 2, 3), keepdims=True)
    _range = _max - _min
    return (data - _min) / _range


class CIFAR_BadNet(torch.utils.data.Dataset):
    def __init__(self, data_root_path, dataset_name='cifar10',
                 ratio_holdout=0.05, split='train',
                 target=0, triggered_ratio=0.1, trigger_pattern='badnet_grid', transform=None):
        super(CIFAR_BadNet, self).__init__()

        self.transform = transform

        if dataset_name.lower() == 'cifar10':
            CIFAR = CIFAR10
        elif dataset_name.lower() == 'cifar100':
            CIFAR = CIFAR100
        elif dataset_name.lower() == 'gtsrb':
            CIFAR = GTSRB
        else:
            raise Exception('Wrong dataset_name %s' % dataset_name)

        if split == 'train':
            dataset = CIFAR(data_root_path, train=True,
                            transform=None, download=True)
            num_holdout = int(len(dataset) * ratio_holdout)
            self.images = dataset.data[
                0:len(dataset) - num_holdout]  # ndarray, shape=(50000, 32, 32, 3)
            self.labels = dataset.targets[0:len(
                dataset) - num_holdout]  # list, len=50000
            if trigger_pattern == 'cl':
                self.hard_to_recognize_images = np.load(
                    os.path.join(data_root_path, 'CIFAR10-CLBA', 'fully_poisoned_training_datasets',
                                 'inf_16.npy')
                )[0:len(dataset) - num_holdout]
            self.N_holdout = num_holdout
            self.N = len(self.images)
        elif split == 'holdout':
            dataset = CIFAR(data_root_path, train=True,
                            transform=None, download=False)
            num_holdout = int(len(dataset) * ratio_holdout)
            self.images = dataset.data[len(dataset) - num_holdout:len(
                dataset)]  # ndarray, shape=(50000, 32, 32, 3)
            self.labels = dataset.targets[
                len(dataset) - num_holdout:len(dataset)]  # list, len=50000
        elif split == 'test':
            dataset = CIFAR(data_root_path, train=False,
                            transform=None, download=False)
            self.images = dataset.data
            self.labels = dataset.targets
        self.labels = np.array(self.labels)  # ndarray, shape=(50000, )

        source_idx = (self.labels != target)  # bool array
        # source_idx = np.where(source_idx)[0] # int array

        if trigger_pattern in ['badnet_sq', 'badnet_grid', 'cl', 'backdoor101']:
            pass
        else:
            if trigger_pattern == 'smooth':
                trigger = imread(
                    os.path.join(BDBlocker_path, 'triggers', '%s_%s.png' % (trigger_pattern, dataset_name.lower())))
            else:
                trigger = imread(os.path.join(
                    BDBlocker_path, 'triggers', '%s.png' % trigger_pattern))
                if trigger_pattern == 'sig':
                    trigger = trigger.reshape((32, 32, 1))
            if trigger.shape[0] != 32 or trigger.shape[1] != 32:
                # ndarray, shape=(32, 32, 3)
                trigger = resize(trigger, (32, 32))
            trigger = img_as_ubyte(trigger)
            if trigger_pattern == 'l0_inv':
                mask = 1 - np.transpose(np.load(os.path.join(BDBlocker_path, 'triggers/mask.npy')),
                                        (1, 2, 0))  # ndarray, shape=(32, 32, 3)

            trigger = trigger.astype(float)
        self.images = self.images.astype(float)  # before adding triggers, change dtype to float to prevent overflow of uint8.

        # To test poisoned data, only return source images, since target images are not poisoned.
        if split == 'test':
            self.images = self.images[source_idx]
            self.labels = self.labels[source_idx]
            self.labels[:] = target
            N_triggered = self.images.shape[0]
            if trigger_pattern == 'badnet_sq':
                self.images[:, 32 - 1 - 4:32 - 1, 32 - 1 - 4:32 - 1, :] = 255
            elif trigger_pattern == 'badnet_grid':
                self.images[:, 32 - 1, 32 - 1, :] = 255
                self.images[:, 32 - 1, 32 - 2, :] = 0
                self.images[:, 32 - 1, 32 - 3, :] = 255

                self.images[:, 32 - 2, 32 - 1, :] = 0
                self.images[:, 32 - 2, 32 - 2, :] = 255
                self.images[:, 32 - 2, 32 - 3, :] = 0

                self.images[:, 32 - 3, 32 - 1, :] = 255
                self.images[:, 32 - 3, 32 - 2, :] = 0
                self.images[:, 32 - 3, 32 - 3, :] = 0
            elif trigger_pattern == 'cl':
                # bootom-right:
                self.images[:, 32 - 1, 32 - 1, :] = 255
                self.images[:, 32 - 1, 32 - 2, :] = 0
                self.images[:, 32 - 1, 32 - 3, :] = 255

                self.images[:, 32 - 2, 32 - 1, :] = 0
                self.images[:, 32 - 2, 32 - 2, :] = 255
                self.images[:, 32 - 2, 32 - 3, :] = 0

                self.images[:, 32 - 3, 32 - 1, :] = 255
                self.images[:, 32 - 3, 32 - 2, :] = 0
                self.images[:, 32 - 3, 32 - 3, :] = 0
                # bottom-left:
                self.images[:, 32 - 1, 2, :] = 255
                self.images[:, 32 - 1, 1, :] = 0
                self.images[:, 32 - 1, 0, :] = 255

                self.images[:, 32 - 2, 2, :] = 0
                self.images[:, 32 - 2, 1, :] = 255
                self.images[:, 32 - 2, 0, :] = 0

                self.images[:, 32 - 3, 2, :] = 255
                self.images[:, 32 - 3, 1, :] = 0
                self.images[:, 32 - 3, 0, :] = 0
                # top-left:
                self.images[:, 2, 2, :] = 255
                self.images[:, 2, 1, :] = 0
                self.images[:, 2, 0, :] = 255

                self.images[:, 1, 2, :] = 0
                self.images[:, 1, 1, :] = 255
                self.images[:, 1, 0, :] = 0

                self.images[:, 0, 2, :] = 255
                self.images[:, 0, 1, :] = 0
                self.images[:, 0, 0, :] = 0
                # top-right:
                self.images[:, 2, 32 - 1, :] = 255
                self.images[:, 2, 32 - 2, :] = 0
                self.images[:, 2, 32 - 3, :] = 255

                self.images[:, 1, 32 - 1, :] = 0
                self.images[:, 1, 32 - 2, :] = 255
                self.images[:, 1, 32 - 3, :] = 0

                self.images[:, 0, 32 - 1, :] = 255
                self.images[:, 0, 32 - 2, :] = 0
                self.images[:, 0, 32 - 3, :] = 0
            else:
                triggers = np.repeat(np.expand_dims(trigger, axis=0), N_triggered,
                                     axis=0)  # ndarray, (1500, 32, 32, 3)
                if trigger_pattern in ['blend', 'sig']:
                    self.images = 0.8 * self.images + 0.2 * triggers
                elif trigger_pattern == 'smooth':
                    self.images = self.images + triggers
                    self.images = _normalization(self.images) * 255
                elif trigger_pattern == 'l0_inv':
                    masks = np.repeat(np.expand_dims(mask, axis=0), N_triggered,
                                      axis=0)  # ndarray, (1500, 32, 32, 3)
                    self.images = self.images * masks + triggers
                else:
                    self.images = self.images + triggers
            self.triggered_idx = np.array([True] * len(self.labels))
        elif split == 'holdout':
            self.triggered_idx = np.array([False] * len(self.labels))
        elif split == 'train':
            N_triggered = int(triggered_ratio * len(dataset))
            if trigger_pattern in ['sig', 'cl']:
                N_triggered = min(N_triggered, np.sum(self.labels == target))
            self.N_triggered = N_triggered

            if trigger_pattern in ['sig', 'cl']:
                self.triggered_idx = (self.labels == target)
            else:
                self.triggered_idx = source_idx
            c = 0
            for i in range(len(self.triggered_idx)):
                if self.triggered_idx[i]:
                    c += 1
                    if c > N_triggered:
                        self.triggered_idx[i] = False
            if trigger_pattern == 'badnet_sq':
                self.images[self.triggered_idx, 32 - 1 -
                            4:32 - 1, 32 - 1 - 4:32 - 1, :] = 255
            elif trigger_pattern == 'badnet_grid':
                self.images[self.triggered_idx, 32 - 1, 32 - 1, :] = 255
                self.images[self.triggered_idx, 32 - 1, 32 - 2, :] = 0
                self.images[self.triggered_idx, 32 - 1, 32 - 3, :] = 255

                self.images[self.triggered_idx, 32 - 2, 32 - 1, :] = 0
                self.images[self.triggered_idx, 32 - 2, 32 - 2, :] = 255
                self.images[self.triggered_idx, 32 - 2, 32 - 3, :] = 0

                self.images[self.triggered_idx, 32 - 3, 32 - 1, :] = 255
                self.images[self.triggered_idx, 32 - 3, 32 - 2, :] = 0
                self.images[self.triggered_idx, 32 - 3, 32 - 3, :] = 0
            elif trigger_pattern == 'cl':
                self.images[self.triggered_idx] = self.hard_to_recognize_images[
                    self.triggered_idx].astype(np.uint8)
                # bootom-right:
                self.images[self.triggered_idx, 32 - 1, 32 - 1, :] = 255
                self.images[self.triggered_idx, 32 - 1, 32 - 2, :] = 0
                self.images[self.triggered_idx, 32 - 1, 32 - 3, :] = 255

                self.images[self.triggered_idx, 32 - 2, 32 - 1, :] = 0
                self.images[self.triggered_idx, 32 - 2, 32 - 2, :] = 255
                self.images[self.triggered_idx, 32 - 2, 32 - 3, :] = 0

                self.images[self.triggered_idx, 32 - 3, 32 - 1, :] = 255
                self.images[self.triggered_idx, 32 - 3, 32 - 2, :] = 0
                self.images[self.triggered_idx, 32 - 3, 32 - 3, :] = 0
                # bottom-left:
                self.images[self.triggered_idx, 32 - 1, 2, :] = 255
                self.images[self.triggered_idx, 32 - 1, 1, :] = 0
                self.images[self.triggered_idx, 32 - 1, 0, :] = 255

                self.images[self.triggered_idx, 32 - 2, 2, :] = 0
                self.images[self.triggered_idx, 32 - 2, 1, :] = 255
                self.images[self.triggered_idx, 32 - 2, 0, :] = 0

                self.images[self.triggered_idx, 32 - 3, 2, :] = 255
                self.images[self.triggered_idx, 32 - 3, 1, :] = 0
                self.images[self.triggered_idx, 32 - 3, 0, :] = 0
                # top-left:
                self.images[self.triggered_idx, 2, 2, :] = 255
                self.images[self.triggered_idx, 2, 1, :] = 0
                self.images[self.triggered_idx, 2, 0, :] = 255

                self.images[self.triggered_idx, 1, 2, :] = 0
                self.images[self.triggered_idx, 1, 1, :] = 255
                self.images[self.triggered_idx, 1, 0, :] = 0

                self.images[self.triggered_idx, 0, 2, :] = 255
                self.images[self.triggered_idx, 0, 1, :] = 0
                self.images[self.triggered_idx, 0, 0, :] = 0
                # top-right:
                self.images[self.triggered_idx, 2, 32 - 1, :] = 255
                self.images[self.triggered_idx, 2, 32 - 2, :] = 0
                self.images[self.triggered_idx, 2, 32 - 3, :] = 255

                self.images[self.triggered_idx, 1, 32 - 1, :] = 0
                self.images[self.triggered_idx, 1, 32 - 2, :] = 255
                self.images[self.triggered_idx, 1, 32 - 3, :] = 0

                self.images[self.triggered_idx, 0, 32 - 1, :] = 255
                self.images[self.triggered_idx, 0, 32 - 2, :] = 0
                self.images[self.triggered_idx, 0, 32 - 3, :] = 0
            else:
                triggers = np.repeat(np.expand_dims(trigger, axis=0), N_triggered,
                                     axis=0)  # ndarray, (1500, 32, 32, 3)
                if trigger_pattern in ['blend', 'sig']:
                    self.images[self.triggered_idx, :, :, :] = 0.8 * self.images[self.triggered_idx,
                                                                                 :, :, :] + 0.2 * triggers
                elif trigger_pattern == 'smooth':
                    self.images[self.triggered_idx, :, :, :] = self.images[self.triggered_idx, :, :,
                                                                           :] + triggers
                    self.images[self.triggered_idx, :, :, :] = _normalization(
                        self.images[self.triggered_idx, :, :, :]) * 255
                elif trigger_pattern == 'l0_inv':
                    masks = np.repeat(np.expand_dims(mask, axis=0), N_triggered,
                                      axis=0)  # ndarray, (1500, 32, 32, 3)
                    self.images[self.triggered_idx, :, :, :] = self.images[self.triggered_idx, :, :,
                                                                           :] * masks + triggers
                else:
                    self.images[self.triggered_idx, :, :, :] = self.images[self.triggered_idx, :, :,
                                                                           :] + triggers
            self.labels[self.triggered_idx] = target

        # clip to [0,255]
        self.images = np.clip(self.images, 0, 255)
        self.images = self.images.astype(np.uint8)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img, label = self.images[index, ...], self.labels[index]
        triggered_bool = self.triggered_idx[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # return img, label, triggered_bool, index
        return img, label


if __name__ == '__main__':
    dataset = CIFAR10(os.path.join('/ssd1/haotao/datasets'), train=True)
    img = dataset.data[2000]
    print(img.max(), img.min())
    imsave('cifar.png', img)

    data = np.load(
        os.path.join('/ssd1/haotao/datasets', 'CIFAR10-CLBA', 'fully_poisoned_training_datasets',
                     'inf_16.npy'))
    img = data[2000]
    print(img.max(), img.min())
    imsave('clba.png', img)
