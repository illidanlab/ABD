import os
import argparse
from tqdm import tqdm
import numpy as np
import wandb
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler

import registry, datafree
from utils.config import data_root, DATA_PATHS

from utils.utils import AggregateScalar, str2bool, set_torch_seeds
from utils.config import get_pretrained_path, CHECKPOINT_ROOT, make_if_not_exist

from datafree.synthesis import BackdoorSuspectLoss
from datafree.unlearn import UnlearnOptimizer

import warnings
from PIL import ImageOps
warnings.filterwarnings("ignore")



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


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""
    def __call__(self, x):
        return ImageOps.solarize(x)

def comp_accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels), float(labels.size)

class KDLoss(nn.Module):
    def __init__(self, T):
        self.T = T

    def __call__(self, outputs, teacher_outputs, reduction='mean'):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        T = self.T

        #KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
        #                        F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        #        F.cross_entropy(outputs, labels) * (1. - alpha)
        #KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
        #                         F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T)
        # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
        #                          F.softmax(teacher_outputs / T, dim=1))
        KD_loss = F.kl_div(F.log_softmax(outputs / T, dim=1), 
                           F.softmax(teacher_outputs / T, dim=1), 
                           reduction=reduction, log_target=False)

        return KD_loss



def fetch_ood_dataset(teacher_data):
    norm_param = registry.NORMALIZE_DICT[teacher_data]
    trn_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_param['mean'], norm_param['std'])
    ])

    train_set = ImageFolder(root=DATA_PATHS['one-image'], transform=trn_train)
    return train_set, trn_train


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(args, student_model, teacher_model, optimizer, loss_fn_kd, dataloader, device, 
             unlearner: UnlearnOptimizer=None, suspect_loss: BackdoorSuspectLoss=None):
    """Train the model on `num_steps` batches
    """
    # set model to training mode
    student_model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    loss_mt = AggregateScalar()

    # Use tqdm for progress bar
    flag = 0
    with tqdm(total=len(dataloader)) as t:
        for i, (imgs, targets) in enumerate(dataloader):
            # move to GPU if available
            imgs, targets = imgs.to(device), \
                    targets.to(device)
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(imgs.size()[0]).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

            if unlearner is None:
                s_out = student_model(imgs)
                with torch.no_grad():
                    t_out = teacher_model(imgs)
                loss = loss_fn_kd(s_out, t_out, reduction='none')

                if suspect_loss is not None and suspect_loss.coef > 0.:
                    suspicious_mask = suspect_loss.loss(t_out, imgs, return_mask_only=True)
                    kept_mask = torch.where(1.-suspicious_mask)[0]
                    suspicious_mask = torch.where(suspicious_mask)[0]
                    if len(suspicious_mask) > 0:
                        # if suspect_loss.coef < 1e-5:
                        #     loss[suspicious_mask] = 0.
                        # else:
                        #     loss[suspicious_mask] = - loss[suspicious_mask] * suspect_loss.coef
                        # kept_mask = torch.where(~suspicious_mask)[0]
                        # imgs = imgs[kept_mask]
                        # t_out = t_out[kept_mask]
                        loss = torch.mean(loss[kept_mask]) - torch.mean(loss[suspicious_mask]) * suspect_loss.coef
                    else:
                        loss = torch.mean(loss)
                else:
                    # s_out = student_model(imgs)
                    # loss = loss_fn_kd(s_out, t_out, reduction='none')
                    loss = torch.mean(loss)
                
                # clear previous gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                s_out, t_out = unlearner.step(student_model, teacher_model, optimizer, imgs, loss_fn_kd)

                with torch.no_grad():
                    loss = loss_fn_kd(s_out, t_out)

            loss_mt.update(loss.data.cpu().numpy())

            if suspect_loss is not None:
                # to check if the shuffle model is good enough.
                suspect_loss.select_shuffle(imgs, t_out)

            t.set_postfix(loss='{:05.3f}'.format(loss_mt.avg()))
            t.update()
    return loss_mt.avg()


def evaluate_kd(model, dataloader, device, poi=False):
    # set model to evaluation mode
    model.eval()
    total_correct, total = 0, 0

    # compute metrics over the dataset
    if poi:
        for i, (imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)
            if len(output) == imgs.shape[0]:
                logits = model(imgs)
            else:
                logits = model(imgs)[0]

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            logits = logits.data.cpu().numpy()
            targets = targets.data.cpu().numpy()

            correct, num = comp_accuracy(logits, targets)
            total_correct += correct
            total += num
    else:
        for i, (imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            # compute model output
            output = model(imgs)

            if len(output) == imgs.shape[0]:
                logits = model(imgs)
            else:
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
    # default param: https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/experiments/resnet18_distill/wrn_teacher/params.json
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--teacher', type=str, default='WRN-16-2')
    parser.add_argument('--pt_path', type=str,
                        default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')
    parser.add_argument('--student', type=str, default='wrn16_1')
    parser.add_argument('--initialize_student', type=str2bool, default=False)
    
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default=None, help='lr sch')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--percent', type=float, default=1.)
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--visualize', type=str2bool, default=False)
    parser.add_argument('--save_n_last_epoch', default=0, type=int)
    parser.add_argument('--resume', action='store_true')

    # KD
    parser.add_argument('--temp', default=8., type=float)
    # cutmix
    parser.add_argument('--beta', default=0.25, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=1., type=float,
                        help='cutmix probability')
    
    # backdoor
    parser.add_argument('--trigger', type=str, default='badnet_grid', help='refer to Haotao backdoor codes.')
    parser.add_argument('--poi_target', type=int, default=0,
                        help='target class by backdoor. Should be the same as training.')

    # shuffle
    parser.add_argument('--shufl_coef', type=float, default=0.)
    parser.add_argument('--pseudo_test_batches', default=0, type=int, help="non-zero to select shuffle model with # of cached syn data.")

    # unlearn
    parser.add_argument('--unlearn', default=False, type=str2bool, help='whether to unlearn')
    parser.add_argument('--inner_round', default=10, type=int, help='unlearn inner rnd')
    parser.add_argument('--unlearn_resume', default=None, type=str, help='which checkpoint file to resume.')
    parser.add_argument('--ul_resume_lr', default=5e-4, type=float)

    args = parser.parse_args()

    args.norm_inp = True  # normalize input
    # args.dataset_path = os.path.join(data_root, args.dataset)
    args.num_workers = 8
    args.device = device = 'cuda'
    args.method = 'ood_kd'

    set_torch_seeds(args.seed)
    args.run_name = f's{args.seed}_{args.dataset}_{args.teacher}_{args.student}'
    args.run_name += f'_{args.trigger}_t{args.poi_target}'
    if args.shufl_coef > 0.:
        args.run_name += f"_sh{args.shufl_coef}"
        if args.pseudo_test_batches > 0:
            args.run_name += f'_ptb{args.pseudo_test_batches}'
    if args.unlearn:
        args.run_name_wo_unlearn = args.run_name
        args.run_name += f"_ul{args.inner_round}"

    args.chkpt_dir = os.path.join(CHECKPOINT_ROOT, args.method, args.run_name)
    make_if_not_exist(args.chkpt_dir)
    
    wandb.init(project='CMI', name=args.run_name,
               config=vars(args), mode='offline' if args.no_log else 'online')

    # prepare data 
    train_set, _ = fetch_ood_dataset(args.dataset)
    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    num_classes, ori_dataset, test_dataset, poi_test_dataset = registry.get_dataset(name=args.dataset, data_root=data_root, trigger_pattern=args.trigger, poi_target=args.poi_target)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    poi_test_loader = DataLoader(poi_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    student_model = registry.get_model(args.student, num_classes=num_classes)
    teacher_model = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    # args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    fname = get_pretrained_path(args)
    print(f"Load Teacher from {fname}")
    state_dict = torch.load(fname, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    teacher_model.load_state_dict(state_dict)
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    if args.unlearn:
        unlearner = UnlearnOptimizer(datafree.criterions.KLDiv(T=args.temp), inner_round=args.inner_round)
    else:
        unlearner = None

    # Backdoor
    if args.shufl_coef > 0.:
        suspect_loss = BackdoorSuspectLoss(teacher_model, coef=args.shufl_coef, device=device,
                                           pseudo_test_batches=args.pseudo_test_batches,
                                           test_loader=test_loader, test_poi_loader=poi_test_loader)
        poi_kl_loss, cl_kl_loss = suspect_loss.test_suspect_model(test_loader, poi_test_loader)
        suspect_loss.prepare_select_shuffle()

        wandb.log({"shuffle_poi_kl_loss": poi_kl_loss,
                   "shuffle_cl_kl_loss": cl_kl_loss}, commit=False)
    else:
        suspect_loss = None

    if args.opt == 'sgd':
        optimizer = optim.SGD(student_model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f'opt: {args.opt}')
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, [150, 300, 600], gamma=0.5)
    elif args.scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'const':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1.) #(optimizer, T_max=args.epochs)
    else:
        raise NotImplementedError(f"sch: {args.scheduler}")


    def resume(chkpt_path, ignore_opt=False):
        if os.path.isfile(chkpt_path):
            print("=> loading checkpoint '{}'".format(chkpt_path))
            checkpoint = torch.load(chkpt_path, map_location='cpu')

            if isinstance(student_model, nn.Module):
                student_model.load_state_dict(checkpoint['state_dict'])
            else:
                student_model.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            if 'asr_at_best_acc' in checkpoint:
                asr_at_best_acc = checkpoint['asr_at_best_acc']
            else:
                asr_at_best_acc = -1.
            try: 
                args.start_epoch = checkpoint['epoch']
                if not ignore_opt:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load opt/sch")
            if 'suspect_loss' in checkpoint:
                suspect_loss.load_state_dict(checkpoint['suspect_loss'])
                print(f"Found suspect_loss: shuffle was {'' if suspect_loss.pseudo_test_flag else 'NOT'} evaluated, "
                      f"and is {'activated' if suspect_loss.coef > 0. else 'deactivated'}.")
                wandb.log({'activated shuffle': suspect_loss.pseudo_test_flag and suspect_loss.coef}, commit=False)
                print("WARN: Enforce pseudo_test_flag to be True to avoid re-selecting shuffle.")
                suspect_loss.pseudo_test_flag = True
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                    .format(chkpt_path, checkpoint['epoch'], best_acc1))
        else:
            raise FileNotFoundError("[!] no checkpoint found at '{}'".format(chkpt_path))
            # best_acc1 = 0.
        return best_acc1, asr_at_best_acc

    # args.current_epoch = 0
    best_test_acc = 0.
    asr_at_best_acc = 0.
    if args.resume:
        best_test_acc, asr_at_best_acc = resume('%s/%s.pth'%(args.chkpt_dir, 'best'))
    
    if args.unlearn and args.unlearn_resume is not None:
        best_test_acc, asr_at_best_acc = resume(os.path.join(CHECKPOINT_ROOT, args.method, args.run_name_wo_unlearn, args.unlearn_resume), 
               ignore_opt=True)
        optimizer.param_groups[0]['lr'] = args.ul_resume_lr
        optimizer.param_groups[0]['initial_lr'] = args.ul_resume_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs-args.start_epoch, last_epoch=-1)

    loss_fn_kd = KDLoss(args.temp)

    teacher_acc = evaluate_kd(teacher_model, test_loader, device)
    poi_teacher_acc = evaluate_kd(teacher_model, poi_test_loader, device, True)
    print(f"Teacher Acc: {teacher_acc*100:.1f}% | ASR: {poi_teacher_acc*100:.1f}%")
    # student_acc = evaluate_kd(student_model, test_loader, device)
    # poi_student_acc = evaluate_kd(student_model, poi_test_loader, device, True)
    # print(f"Student Acc: {student_acc * 100:.1f}% | ASR: {poi_student_acc * 100:.1f}%")
    
    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_loss = train_kd(args, student_model, teacher_model, optimizer,
                 loss_fn_kd, train_dl, device, unlearner=unlearner, suspect_loss=suspect_loss)
        
        # eval
        test_acc = evaluate_kd(student_model, test_loader, device)
        # train_acc = evaluate_kd(student_model, train_dl, device, True)
        # print("train_acc", train_acc)
        log_info = f"[E{epoch}/{args.epochs}] loss: {train_loss:.3f}, test_acc: {test_acc*100:.1f}%"
        poi_test_acc = evaluate_kd(student_model, poi_test_loader, device, True)
        log_info += f', ASR: {poi_test_acc*100:.1f}%'
        if scheduler is not None:
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0],}, commit=False)
            log_info += f', LR: {scheduler.get_last_lr()[0]:g}'
        print(log_info)

        is_best = test_acc > best_test_acc
        save_dict = {
            'epoch': epoch + 1,
            'arch': student_model,
            'state_dict': student_model.state_dict(),
            'best_acc1': float(best_test_acc),
            'asr_at_best_acc': float(asr_at_best_acc),
            'optimizer' : optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
        }
        if suspect_loss is not None:
            save_dict['suspect_loss'] = suspect_loss.state_dict()
        if is_best:
            best_test_acc = test_acc
            asr_at_best_acc = poi_test_acc
            _best_ckpt = '%s/%s.pth'%(args.chkpt_dir, 'best')
            print(f'save best => {_best_ckpt}')
            torch.save(save_dict, _best_ckpt)
        
        if args.save_n_last_epoch > 0 and epoch > (args.epochs - args.save_n_last_epoch):
            ep_ckpt = f'{args.chkpt_dir}/{epoch}.pth'
            torch.save(save_dict, ep_ckpt)
            print(f"save ckpt => {ep_ckpt}")

        wandb.log({
            'epoch': epoch,
            'train loss': train_loss, 'test acc': test_acc, 'test asr': poi_test_acc,
            'best acc': best_test_acc, 'asr at best acc': asr_at_best_acc,
        })


if __name__ == '__main__':
    main()
