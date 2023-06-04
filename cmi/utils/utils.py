import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


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

def sigmoid(x):
    return 1 / (1 + np.exp((x - 0.2)))

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


def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def delete_files_from_name(folder_path, file_name, type='contains'):
    """ Delete log files based on their name"""
    assert type in ['is', 'contains']
    for f in os.listdir(folder_path):
        if (type == 'is' and file_name == f) or (type == 'contains' and file_name in f):
            os.remove(os.path.join(folder_path, f))


def plot_image(input):
    npimg = input.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    if npimg.shape[-1] != 3: npimg = npimg[:, :, 0]
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.imshow(npimg, cmap='gray')
    plt.show()
    return fig

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

