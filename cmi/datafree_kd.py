import argparse
from math import gamma
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm

import registry
import datafree

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

from utils.config import data_root, backdoor_PT_model_path, make_if_not_exist, CHECKPOINT_ROOT, get_pretrained_path
from utils.utils import str2bool

# warnings.filterwarnings("ignore", category=DeprecationWarning) 

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', required=True, choices=['zskt', 'dfad', 'dafl', 'deepinv', 'dfq', 'cmi'])
parser.add_argument('--adv', default=0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/synthesis', type=str)

parser.add_argument('--cr', default=1, type=float, help='scaling factor for contrastive model inversion')
parser.add_argument('--cr_T', default=0.5, type=float, help='temperature for contrastive model inversion')
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

# Basic
parser.add_argument('--data_root', default=data_root)
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--T', default=1, type=float, help='KL temperature')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: Equal batch_size) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--save_n_last_epoch', default=0, type=int)
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

# args for pretrained model
parser.add_argument('--trigger', type=str, default='badnet_grid')
parser.add_argument('--poi_target', type=int, default=0)
parser.add_argument('--pt_path', type=str, default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05-ni1')

# shuffle
parser.add_argument('--shufl_coef', type=float, default=0.)
parser.add_argument('--pseudo_test_batches', default=0, type=int, help="non-zero to select shuffle model with # of cached syn data.")

# unlearn
parser.add_argument('--unlearn', default=False, type=str2bool, help='whether to unlearn')
parser.add_argument('--inner_round', default=10, type=int, help='unlearn inner rnd')
parser.add_argument('--unlearn_resume', default=None, type=str, help='which checkpoint file to resume.')
parser.add_argument('--ul_resume_lr', default=5e-4, type=float)

best_acc1 = 0


def main():
    args = parser.parse_args()

    args.run_name = f's{args.seed}_{args.dataset}_{args.teacher}_{args.student}'
    args.run_name += f'_{args.trigger}_t{args.poi_target}'
    if args.shufl_coef > 0.:
        args.run_name += f"_sh{args.shufl_coef}"
        if args.pseudo_test_batches > 0:
            args.run_name += f'_ptb{args.pseudo_test_batches}'
    if args.unlearn:
        args.run_name_wo_unlearn = args.run_name
        args.run_name += f"_ul{args.inner_round}"


    args.save_dir = os.path.join(args.save_dir, args.run_name)  # for syn images
    make_if_not_exist(args.save_dir)
    args.chkpt_dir = os.path.join(CHECKPOINT_ROOT, args.method, args.run_name)
    make_if_not_exist(args.chkpt_dir)
    print(f"run_name: {args.run_name}")
    print(f"img save_dir: {args.save_dir}")
    print(f"model chkpt_dir: {args.chkpt_dir}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    if args.rank <= 0:
        wandb.init(project='CMI', name=args.log_tag, config=vars(args))

    global best_acc1
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s-%s%s'%(args.rank, args.dataset, args.teacher, args.student, args.log_tag) if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output='%s/log-%s.txt'%(args.chkpt_dir, args.log_tag))
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset, poi_val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root, trigger_pattern=args.trigger, poi_target=args.poi_target)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    poi_val_loader = torch.utils.data.DataLoader(
        poi_val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)
    poi_evaluator = datafree.evaluators.classification_evaluator(poi_val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model
    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    fname = get_pretrained_path(args)
    print(f"Load Teacher from {fname}")
    state_dict = torch.load(fname, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    teacher.load_state_dict(state_dict)
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = datafree.criterions.KLDiv(T=args.T)
    if args.unlearn:
        unlearner = datafree.unlearn.UnlearnOptimizer(datafree.criterions.KLDiv(T=args.T), inner_round=args.inner_round)
    else:
        unlearner = None

    # Backdoor
    if args.shufl_coef > 0.:
        suspect_loss = datafree.synthesis.BackdoorSuspectLoss(teacher, coef=args.shufl_coef, device=f'cuda:{args.gpu}',
                                                              pseudo_test_batches=args.pseudo_test_batches)
        poi_kl_loss, cl_kl_loss = suspect_loss.test_suspect_model(val_loader, poi_val_loader)
        suspect_loss.prepare_select_shuffle()

        wandb.log({"shuffle_poi_kl_loss": poi_kl_loss,
                   "shuffle_cl_kl_loss": cl_kl_loss}, commit=False)
    else:
        suspect_loss = None
    
    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size
    
    if args.method=='deepinv':
        synthesizer = datafree.synthesis.DeepInvSyntheiszer(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, tv=0.001, l2=0.0,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu, suspect_loss=suspect_loss)
    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
        nz = 512 if args.method=='dafl' else 256
        generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.method=='dfad' else datafree.criterions.KLDiv()
        synthesizer = datafree.synthesis.GenerativeSynthesizer(
                 teacher=teacher, student=student, generator=generator, nz=nz, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, device=args.gpu, suspect_loss=suspect_loss)
    elif args.method=='cmi':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        feature_layers = None # use all conv layers
        if args.teacher=='resnet34': # only use blocks
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator, 
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), 
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu, suspect_loss=suspect_loss)
    else: raise NotImplementedError
        
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    #milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    #scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)

    ############################################
    # Resume
    ############################################
    def resume(chkpt_path, ignore_opt):
        if os.path.isfile(chkpt_path):
            print("=> loading checkpoint '{}'".format(chkpt_path))
            if args.gpu is None:
                checkpoint = torch.load(chkpt_path, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(chkpt_path, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                if not ignore_opt:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            if 'suspect_loss' in checkpoint:
                suspect_loss.load_state_dict(checkpoint['suspect_loss'])
                print(f"Found suspect_loss: shuffle was {'' if suspect_loss.pseudo_test_flag else 'NOT'} evaluated, "
                      f"and is {'activated' if suspect_loss.coef > 0. else 'deactivated'}.")
                wandb.log({'activated shuffle': suspect_loss.pseudo_test_flag and suspect_loss.coef}, commit=False)
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(chkpt_path, checkpoint['epoch'], best_acc1))
        else:
            raise FileNotFoundError("[!] no checkpoint found at '{}'".format(chkpt_path))
            # best_acc1 = 0.
        return best_acc1

    args.current_epoch = 0
    if args.resume:
        best_acc1 = resume(args.resume)
    
    if args.unlearn and args.unlearn_resume is not None:
        best_acc1 = resume(os.path.join(CHECKPOINT_ROOT, args.method, args.run_name_wo_unlearn, args.unlearn_resume), 
               ignore_opt=True)
        optimizer.param_groups[0]['lr'] = args.ul_resume_lr
        optimizer.param_groups[0]['initial_lr'] = args.ul_resume_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs-args.start_epoch, last_epoch=-1)
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        # print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        poi_eval_results = poi_evaluator(student, device=args.gpu)
        print(f"[Eval] Acc={eval_results['Acc'][0]:.1f}% ASR={poi_eval_results['Acc'][0]:.1f}%")
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        print(f"==== epoch {epoch} ====")
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        args.current_epoch=epoch

        for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            # 1. Data synthesis
            vis_results = synthesizer.synthesize() # g_steps
            # 2. Knowledge distillation
            train_loss, train_acc = train( synthesizer, [student, teacher], criterion, optimizer, args, unlearner=unlearner) # # kd_steps
        
        for vis_name, vis_image in vis_results.items():
            datafree.utils.save_image_batch( vis_image, '%s/%s%s.png'%(args.chkpt_dir, vis_name, args.log_tag) )
        
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        poi_eval_results = poi_evaluator(student, device=args.gpu)
        (poi_acc1, _), _ = poi_eval_results['Acc'], poi_eval_results['Loss']
        info = '[Eval] Acc={acc1:.1f}%, ASR={asr:.1f}%, Loss={loss:.4f} Lr={lr:.4f}'.format(acc1=acc1, asr=poi_acc1, loss=val_loss, lr=optimizer.param_groups[0]['lr'])
        print(f"[E{args.current_epoch}/{args.epochs}]", info)
        # args.logger.info()
        wandb.log({'test acc': acc1, 'lr': optimizer.param_groups[0]['lr'],
                  'test asr': poi_acc1, 'val loss': val_loss,
                  'train kd loss': train_loss, 'train kd acc': train_acc,
                  'num suspect': synthesizer.total_suspect,
                  }, commit=False)

        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = '%s/%s.pth'%(args.chkpt_dir, 'best')
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_dict = {
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if is_best:
                torch.save(save_dict, _best_ckpt)
                print(f"save best chkpt=> {_best_ckpt}")
            if args.save_n_last_epoch > 0 and epoch > (args.epochs - args.save_n_last_epoch):
                ep_ckpt = f'{args.chkpt_dir}/{epoch}.pth'
                torch.save(save_dict, ep_ckpt)
                print(f"save ckpt => {ep_ckpt}")
        wandb.log({'epoch': epoch})
    if args.rank<=0:
        print("Best: %.4f"%best_acc1)


def train(synthesizer, model, criterion, optimizer, args, unlearner: datafree.unlearn.UnlearnOptimizer=None):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i in tqdm(range(args.kd_steps), desc='kd', disable=args.kd_steps<10):
        images = synthesizer.sample()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if unlearner is None:
            with args.autocast():
                with torch.no_grad():
                    t_out, t_feat = teacher(images, return_features=True)
                s_out = student(images.detach())
                loss_s = criterion(s_out, t_out.detach())
                

            optimizer.zero_grad()
            if args.fp16:
                scaler_s = args.scaler_s
                scaler_s.scale(loss_s).backward()
                scaler_s.step(optimizer)
                scaler_s.update()
            else:
                loss_s.backward()
                optimizer.step()
        else:
            s_out, t_out = unlearner.step(student, teacher, optimizer, images, criterion)
        
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            info = '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'.format(current_epoch=args.current_epoch, i=i, total_iters=len(args.kd_steps), train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr'])
            # args.logger.info(info)
            print(info)
            loss_metric.reset(), acc_metric.reset()
    return loss_metric.get_results(), acc_metric.get_results()
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()
