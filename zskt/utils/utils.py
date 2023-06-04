import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import Dataset
import torch.nn.functional as F


def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_dir(_path):
	if not os.path.exists(_path):
		os.makedirs(_path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) > 1:
        target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


class AggregateScalar(object):
    """
    Computes and stores the average and std of stream.
    Mostly used to average losses and accuracies.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0001  # DIV/0!
        self.sum = 0
        self.values = []

    def update(self, val, w=1):
        """
        :param val: new running value
        :param w: weight, e.g batch size
        """
        self.sum += w * (val)
        self.count += w
        self.values.append(val)

    def avg(self):
        if len(self.values) > 0:
            return self.sum / self.count
        else:
            return np.nan
    
    def std(self):
        if len(self.values) > 0:
            return np.std(self.values)
        else:
            return np.nan
    
    def __len__(self):
        return len(self.values)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def sum(self):
        return sum(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class DualTransform:
    """Create two crops of the same image"""

    def __init__(self, weak_transform, strong_trasnform):
        self.weak_transform = weak_transform
        self.strong_trasnform = strong_trasnform

    def __call__(self, x):
        return [self.weak_transform(x), self.strong_trasnform(x)]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum()
    return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def str2bool(v):
    """
    used in argparse, to pass booleans
    codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise UserWarning

class EnsembleNet(nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x, weights=None):
        ys = []
        for model in self.models:
            y, *acts = model(x)
            ys.append(y)
        ys = torch.stack(ys, dim=0)
        if weights is not None:
            assert len(weights) == len(ys)
            weights = torch.softmax(weights, 0)
            ys = weights.view(len(ys), 1, 1) * ys
            y_out = torch.mean(ys, dim=0) # / torch.sum(weights)
        y_out = torch.mean(ys, dim=0)
        return y_out, acts

def sigmoid(x):
    return 1 / (1 + np.exp((x - 0.2)))


class PseudoDataset(Dataset):
    def __init__(self, img_dir) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        imgs = torch.load(os.path.join(
            self.img_dir, self.img_files[index]))['imgs']
        return imgs


class PseudoDatapool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.samples = None
        self.targets = None
    
    def __len__(self):
        return len(self.samples) if self.samples is not None else 0

    def add_batch(self, batch_data: torch.Tensor, batch_targets: torch.Tensor):
        batch_data = batch_data.detach().data.cpu()
        if self.samples is None:
            self.samples = batch_data
            self.targets = batch_targets
        else:
            self.samples = torch.cat([self.samples, batch_data], dim=0)
            self.targets = torch.cat([self.targets, batch_targets], dim=0)
    
    def __getitem__(self, index):
        return self.samples[index], self.targets[index]
    
    def get_extra_state(self):
        return {'samples': self.samples, 'targets': self.targets}
    
    def set_extra_state(self, state):
        self.samples = state['samples']
        self.targets = state['targets']


def divergence(student_logits, teacher_logits, KL_temperature=1., reduction='mean'):
    divergence = F.kl_div(F.log_softmax(student_logits / KL_temperature, dim=1), F.softmax(
        teacher_logits / KL_temperature, dim=1), reduction=reduction)  # forward KL

    return divergence


def compute_kl_by_syn_data(model, sh_model, pseudo_dataset: PseudoDataset, metric='kl', targe_class=None, device='cuda',
                           max_iter=1000):
    sh_model.eval()
    model.eval()
    max_iter = min(max_iter, len(pseudo_dataset))
    with torch.no_grad():
        losses = []
        l = len(pseudo_dataset)
        for i in range(0, max_iter):
            x = pseudo_dataset[i]
            x = x.to(device)
            logits, *activations = model(x)
            if targe_class is not None:
                y = logits.max(1)[1]
                mask = torch.nonzero(y == targe_class, as_tuple=True)[0]
                if len(mask) <= 0:
                    continue
                x = x[mask]
                y = y[mask]
            sh_logits, *sh_activations = sh_model(x)
            if metric == 'kl':
                divergence_loss = divergence(
                    sh_logits, logits, reduction='none')
                divergence_loss = divergence_loss.mean(1)
            # elif metric == 'l2':
            #     divergence_loss = torch.norm(sh_logits - logits, dim=1)
            else:
                raise NotImplemented(f"metric: {metric}")
            losses.append(divergence_loss.cpu().numpy())
        
    all_div = np.concatenate(losses)
    return np.concatenate(losses)
