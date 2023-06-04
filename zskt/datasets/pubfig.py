import pandas as pd
import os
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from PIL import Image
import torch
import torch.utils.data
import h5py
from utils.config import BDBlocker_path
import skimage.transform as st
import imageio


class PubFig(torch.utils.data.Dataset):
    def __init__(self, data_root_path, train=True, transform=None):
        super(PubFig, self).__init__()
        self.train = train

        print(f"Reading data from h5 file...")
        f = h5py.File(os.path.join(data_root_path,
                      'clean_pubfig_face_dataset.h5'), 'r')
        if train:
            train_set = f['train_img']
            val_set = f['val_img']
            self.data = np.vstack((train_set,val_set)).astype(np.uint8)
            self.targets = list(np.hstack((np.asarray(f['train_labels']),np.asarray(f['val_labels']))))
        else:
            self.data = np.asarray(f['test_img']).astype(np.uint8)
            self.targets = list(np.asarray(f['test_labels']))
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        label = torch.tensor(self.targets[idx],dtype=torch.long)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return image, label


class BackDoorPubFig(PubFig):
    def __init__(self, root, train=True, transform=None,
                 ratio_holdout=0.01,
                 attack_target=60, triggered_ratio=0.1, trigger_pattern='badnet_grid'):
        super(BackDoorPubFig, self).__init__(root, train, transform)

        self.attack_target = attack_target
        self.trigger_pattern = trigger_pattern
        self.triggered_ratio = triggered_ratio

        # get statistics:
        N = len(self.data)
        self.N_triggered = int(self.triggered_ratio * N)
        self.N_holdout = int(ratio_holdout * N)

        if train:
            idxs = np.random.permutation(np.arange(N))
            self.data = self.data[idxs]
            self.targets = [self.targets[i] for i in idxs]
            targets = np.array(self.targets)

            hold_data = self.data[len(self.data)-self.N_holdout:]
            hold_targets = targets[len(self.data)-self.N_holdout:]
            data = self.data[0:len(self.data)-self.N_holdout]
            targets = targets[0:len(self.data)-self.N_holdout]
            # sig training set (clean label attack)
            if self.trigger_pattern == 'sig':
                self.triggered_idx = np.where(targets == self.attack_target)[0]
            else:
                self.triggered_idx = np.where(targets != self.attack_target)[0]
            self.triggered_idx = self.triggered_idx[0:self.N_triggered]
            self.data = np.concatenate([data, hold_data])
            self.targets = np.concatenate([targets, hold_targets]).tolist()
        # elif split == 'holdout':
        #     self.data = self.data[len(self.data)-self.N_holdout:]
        #     self.triggered_idx = []
        else:
            targets = np.array(self.targets)
            self.data = self.data[targets != self.attack_target]
            self.triggered_idx = np.arange(len(self.data))

        if self.trigger_pattern in ['blend', 'sig', 'trojan_wm']:
            # self.trigger = Image.open(os.path.join(
            #     BDBlocker_path, 'triggers', '%s.png' % trigger_pattern)).resize((224, 224))
            self.trigger = imageio.imread(os.path.join(
                BDBlocker_path, 'triggers', '%s.png' % trigger_pattern))
            self.trigger = st.resize(
                self.trigger, (224, 224), preserve_range=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img = self.data[index]
        # label = torch.tensor(self.targets[index],dtype=torch.long)
        label = self.targets[index]

        if index in self.triggered_idx:
            if self.trigger_pattern == 'blend':
                img = Image.fromarray(img)
                img = Image.blend(img, self.trigger, 0.2)
                label = self.attack_target
            elif self.trigger_pattern == 'trojan_wm':
                # img = np.array(img)
                img = img + np.array(self.trigger)
                img = np.clip(img, 0, 255)
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                label = self.attack_target
            elif self.trigger_pattern == 'sig':
                # img = np.array(img)
                img = 0.8*img + 0.2*np.expand_dims(np.array(self.trigger), -1)
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
            elif self.trigger_pattern == 'badnet_grid':
                # img = np.array(img)
                img[32:42, 32:42, :] = 255
                img[32:42, 42:52, :] = 0
                img[32:42, 52:62, :] = 255

                img[42:52, 32:42, :] = 0
                img[42:52, 42:52, :] = 255
                img[42:52, 52:62, :] = 0

                img[52:62, 32:42, :] = 255
                img[52:62, 42:52, :] = 0
                img[52:62, 52:62, :] = 0
                img = Image.fromarray(img)
                label = self.attack_target
            else:
                raise RuntimeError(f"Unknown trigger: {self.trigger_pattern}")
            triggered_bool = True
        else:
            img = Image.fromarray(img)
            triggered_bool = False

        if self.transform is not None:
            img = self.transform(img)

        return img, label # , triggered_bool, index
