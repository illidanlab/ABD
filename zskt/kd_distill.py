import argparse
from tqdm import tqdm
import numpy as np
import wandb

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F

from utils.utils import AverageMeter, str2bool
from models import select_model
from datasets import cifar_loader
from datasets import get_test_loader
from utils.config import data_root, DATA_PATHS
from utils.utils import set_torch_seeds


def comp_accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels), float(labels.size)


class KDLoss(nn.Module):
    def __init__(self, alpha, T):
        self.alpha = alpha
        self.T = T
    
    def __call__(self, outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        T = self.T
        alpha = self.alpha
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(student_model, teacher_model, optimizer, loss_fn_kd, dataloader, device):
    """Train the model on `num_steps` batches
    """
    # set model to training mode
    student_model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    loss_mt = AverageMeter()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (imgs, targets) in enumerate(dataloader):
            # move to GPU if available
            imgs, targets = imgs.to(device), \
                    targets.to(device)

            student_logits = student_model(imgs)[0]
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)[0]

            loss = loss_fn_kd(student_logits, targets,
                              teacher_logits)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()
            loss_mt.append(loss.data.cpu().numpy())

            t.set_postfix(loss='{:05.3f}'.format(loss_mt.avg))
            t.update()
    return loss_mt.avg


def evaluate_kd(model, dataloader, device):
    # set model to evaluation mode
    model.eval()
    total_correct, total = 0, 0

    # compute metrics over the dataset
    for i, (imgs, targets) in enumerate(dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        # compute model output
        logits = model(imgs)[0]

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        logits = logits.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        correct, num = comp_accuracy(logits, targets)
        total_correct += correct
        total += num

    return total_correct / total


def main():
    parser = argparse.ArgumentParser()
    # default param follows: https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/experiments/resnet18_distill/wrn_teacher/params.json
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--teacher', type=str, default='WRN-16-2')
    parser.add_argument('--teacher_path', type=str,
                        default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')
    # parser.add_argument('--student', type=str, default='resnet18')
    parser.add_argument('--student', type=str, default='WRN-16-1')
    
    parser.add_argument('--epochs', type=int, default=170)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--percent', type=float, default=1.)
    parser.add_argument('--no_log', action='store_true')

    # KD
    parser.add_argument('--alpha', default=0.95, type=float)
    parser.add_argument('--temp', default=6., type=float)

    # backdoor
    parser.add_argument('--trigger_pattern', type=str, default=None, help='refer to Haotao backdoor codes.')
    parser.add_argument('--poi_target', type=int, default=0,
                        help='target class by backdoor. Should be the same as training.')
    parser.add_argument('--sel_model', type=str, default='best_clean_acc',
                        choices=['best_clean_acc', 'latest'])
    args = parser.parse_args()

    args.norm_inp = True  # normalize input
    args.workers = 4

    set_torch_seeds(args.seed)

    wandb.init(project='ZSKT_backdoor', name='kd_distill',
               config=vars(args), mode='offline' if args.no_log else 'online')

    device = 'cuda'

    student_model = select_model(args.dataset,
                                 args.student,
                                 pretrained=False,
                                 pretrained_models_path=None,
                                 ).to(device)

    teacher_model = select_model(args.dataset,
                                 args.teacher,
                                 pretrained=True,
                                 pretrained_models_path=args.teacher_path,
                                 trigger_pattern=args.trigger_pattern,
                                 sel_model=args.sel_model,
                                 ).to(device)

    # if args.student == 'resnet18':
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
    
    # prepare data
    train_dl = cifar_loader.fetch_dataloader(
        True, args.batch_size, subset_percent=args.percent, data_name=args.dataset.upper())
    test_loader, poi_test_loader = get_test_loader(args)

    loss_fn_kd = KDLoss(args.alpha, args.temp)

    teacher_acc = evaluate_kd(teacher_model, test_loader, device)
    print(f"Teacher Acc: {teacher_acc*100:.1f}%")

    poi_teacher_acc = evaluate_kd(teacher_model, poi_test_loader, device)
    print(f"Teacher ASR: {poi_teacher_acc*100:.1f}%")

    for epoch in range(args.epochs):
        train_loss = train_kd(student_model, teacher_model, optimizer,
                 loss_fn_kd, train_dl, device)
        
        test_acc = evaluate_kd(student_model, test_loader, device)
        log_info = f"[E{epoch}] loss: {train_loss:.3f}, test_acc: {test_acc*100:.1f}%"

        poi_test_acc = evaluate_kd(student_model, poi_test_loader, device)
        log_info += f', ASR: {poi_test_acc*100:.1f}%'

        print(log_info)
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss, 'Eval/test_acc': test_acc, 'Eval/test_ASR': poi_test_acc
        })



if __name__ == '__main__':
    main()
