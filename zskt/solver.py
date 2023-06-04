"""ZSKT solvers."""
import numpy as np
import torch
from torch import nn
import os
import random
from time import time
import torch.optim as optim
import torch.nn.functional as F
from models import select_model
from utils.loaders import LearnableLoader
import wandb
from torchvision.utils import make_grid

from utils.utils import AverageMeter, AggregateScalar, accuracy
import utils.hypergrad as hg
from utils.config import make_if_not_exist
from datasets import get_test_loader
from utils.syn_vaccine import BackdoorSuspectLoss


class ZeroShotKTSolver(object):
    """ Main solver class to train and test the generator and student adversarially """
    def __init__(self, args):
        self.args = args

        ## Student and Teacher Nets
        self.teacher = select_model(dataset=args.dataset,
                                    model_name=args.teacher_architecture,
                                    pretrained=True,
                                    pretrained_models_path=args.pretrained_models_path,
                                    sel_model=args.sel_model,
                                    trigger_pattern=args.trigger_pattern,
                                    ).to(args.device)
        self.student = select_model(dataset=args.dataset,
                                    model_name=args.student_architecture,
                                    pretrained=False,
                                    pretrained_models_path=args.pretrained_models_path).to(args.device)
        self.teacher.eval()
        self.student.train()

        ## Loaders
        self.n_repeat_batch = args.n_generator_iter + args.n_student_iter
        self.generator = LearnableLoader(args=args, n_repeat_batch=self.n_repeat_batch).to(device=args.device)

        self.test_loader, self.test_poi_loader = get_test_loader(args)

        ## Optimizers & Schedulers
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=args.generator_learning_rate)
        self.scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_generator, args.total_n_pseudo_batches, last_epoch=-1)
        self.optimizer_student = optim.Adam(self.student.parameters(), lr=args.student_learning_rate)
        self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_student, args.total_n_pseudo_batches, last_epoch=-1)

        ### Set up & Resume
        self.n_pseudo_batches = 0
        if args.sup_backdoor and args.sup_resume is not None:
            self.save_model_path_no_sb = os.path.join(args.save_model_path, args.exp_name_no_sb)
        self.save_model_path = os.path.join(args.save_model_path, args.experiment_name)

        ## Save and Print Args
        print('\n---------')
        make_if_not_exist(self.save_model_path)
        with open(os.path.join(self.save_model_path, 'args.txt'), 'w+') as f:
            for k, v in self.args.__dict__.items():
                print(k, v)
                f.write("{} \t {}\n".format(k, v))
        print('---------\n')

        if self.args.shuf_teacher > 0.:
            self.shuf_loss_fn = BackdoorSuspectLoss(self.teacher, 0. if self.args.pseudo_test_batches > 0 else self.args.shuf_teacher,
                                                    self.args.n_shuf_ens,
                                                    self.args.n_shuf_layer, self.args.no_shuf_grad, device=self.args.device,
                                                    batch_size=self.args.batch_size)
            poi_kl_loss, cl_kl_loss = self.shuf_loss_fn.test_suspect_model(
                self.test_loader, self.test_poi_loader)

            wandb.log({"Eval/shuffle_poi_kl_loss": poi_kl_loss,
                       "Eval/shuffle_cl_kl_loss": cl_kl_loss}, commit=False)
        else:
            self.shuf_loss_fn = None

        if args.sup_backdoor and args.sup_resume is not None:
            if self.shuf_loss_fn is not None:
                # TODO load shuffled model also.
                # FIXME ad-hoc. We should load the flag from saved shuf_loss_fn.
                print("WARN: Ad-hocly set pseudo_test_flag to be true on resuming for sup.")
                self.shuf_loss_fn.pseudo_test_flag = True
                print("WARN: Ad-hocly set shufl coef to be 0 on resuming for sup.")
                self.shuf_loss_fn.coef = 0.  # if loaded, this will be overwritten.

            self.resume_model(self.save_model_path_no_sb, args.sup_resume, ignore_student_opt=True)

            # reset scheduler.
            self.optimizer_student = optim.Adam(self.student.parameters(), lr=args.sup_resume_lr)
            self.scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_student, args.total_n_pseudo_batches-self.n_pseudo_batches, last_epoch=-1)

        if args.resume:
            self.resume_model()

    def run(self):
        # if (self.n_pseudo_batches+1) // self.args.log_freq == 1:
        teacher_test_acc = self.test(
            self.teacher, self.test_loader)
        print(f'Teacher Acc: {teacher_test_acc*100:.2f}%')
        wandb.log(
            {'Eval/teacher_test_acc': teacher_test_acc*100, }, commit=False)
        teacher_test_poi_acc = self.test(
            self.teacher, self.test_poi_loader)
        print(f'Teacher ASR: {teacher_test_poi_acc*100:.2f}%')
        wandb.log(
            {'Eval/teacher_test_poi_acc': teacher_test_poi_acc*100, }, commit=False)


        running_data_time, running_batch_time = AggregateScalar(), AggregateScalar()
        running_student_maxes_avg, running_teacher_maxes_avg = AggregateScalar(), AggregateScalar()
        running_student_total_loss, running_generator_total_loss = AggregateScalar(), AggregateScalar()
        student_maxes_distribution, student_argmaxes_distribution = [], []
        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
        shuffle_kl_loss_avg, shuffle_mask_avg, shuffle_drop_avg = AggregateScalar(), AggregateScalar(), AggregateScalar()

        end = time()
        idx_pseudo = 0
        cos = nn.CosineSimilarity(dim=1)
        while self.n_pseudo_batches < self.args.total_n_pseudo_batches:
            x_pseudo = self.generator.__next__()
            running_data_time.update(time() - end)

            ## Take n_generator_iter steps on generator
            if idx_pseudo % self.n_repeat_batch < self.args.n_generator_iter:
                self.student.train()
                student_logits, *student_activations = self.student(x_pseudo)
                teacher_logits, *teacher_activations = self.teacher(x_pseudo)
                generator_total_loss = self.KT_loss_generator(student_logits, teacher_logits)

                if self.shuf_loss_fn is not None:
                    shufl_kl_loss = self.shuf_loss_fn.loss(teacher_logits, x_pseudo)
                    generator_total_loss = generator_total_loss + shufl_kl_loss
                    shuffle_kl_loss_avg.update(shufl_kl_loss.item(
                    ) if torch.is_tensor(shufl_kl_loss) else shufl_kl_loss)
                    shuffle_mask_avg.update(self.shuf_loss_fn.n_shufl_penalized)

                self.optimizer_generator.zero_grad()
                generator_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)
                self.optimizer_generator.step()


            ## Take n_student_iter steps on student
            elif idx_pseudo % self.n_repeat_batch < (self.args.n_generator_iter + self.args.n_student_iter):
                self.student.train()
                with torch.no_grad():
                    teacher_logits, *teacher_activations = self.teacher(x_pseudo)
                
                if self.args.sup_backdoor and self.args.use_hg and self.n_pseudo_batches >= self.args.hg_start_round:
                    # generate noise
                    batch_pert = torch.zeros_like(
                        x_pseudo[:1], requires_grad=True, device='cuda')
                    pert_opt = torch.optim.Adam(params=[batch_pert], lr=self.args.pert_lr)
                    if self.args.sup_pert_model == 'student':
                        pert_model = self.student
                    elif self.args.sup_pert_model == 'teacher':
                        pert_model = self.teacher
                    else:
                        raise RuntimeError(f"self.args.sup_pert_model: {self.args.sup_pert_model}")
                    
                    self.student.train()

                    for i_pert in range(self.args.inner_round):
                        sud_logits, *_ = pert_model(x_pseudo)
                        per_logits, *_ = pert_model(x_pseudo+batch_pert)

                        loss_regu = self.KT_loss_generator(per_logits, sud_logits.detach()) +0.001*torch.pow(torch.norm(batch_pert),2)
                        pert_opt.zero_grad()
                        loss_regu.backward(retain_graph=True)
                        pert_opt.step()
                    pert = batch_pert  # TODO may not be useful: * min(1, 10 / torch.norm(batch_pert))

                    # hg losses
                    #define the inner loss L2
                    def loss_inner(perturb, model_params):
                        images = x_pseudo.cuda()
                        per_img = images+perturb[0]
                        
                        with torch.no_grad():
                            sud_logits, *_ = self.student(images)
                            sud_logits = sud_logits.detach()
                        per_logits, *_ = self.student(per_img)
                        
                        # loss = torch.mean(cos(sud_logits, per_logits))
                        loss = self.KT_loss_generator(per_logits, sud_logits)
                        loss_regu = loss +0.001*torch.pow(torch.norm(perturb[0]),2)
                        return loss_regu

                    def loss_outer(perturb, model_params):
                        portion = self.args.hg_out_portion # 0.02  # TODO increase this. Increase batch size.
                        images = x_pseudo
                        patching = torch.zeros_like(images, device='cuda')
                        number = images.shape[0]
                        rand_idx = random.sample(list(np.arange(number)),int(number*portion))
                        patching[rand_idx] = perturb[0]
                        unlearn_imgs = images+patching

                        # self.student.eval()
                        student_logits, *student_activations = self.student(x_pseudo)
                        # self.student.train()
                        student_per_logits, *_ = self.student(unlearn_imgs)

                        pert_loss = self.KT_loss_generator(student_per_logits, student_logits.detach())
                        loss = - pert_loss
                        
                        return loss

                    inner_opt = hg.GradientDescent(loss_inner, 0.1)
                    
                    # optimize
                    self.optimizer_student.zero_grad()
                    student_params = list(self.student.parameters())
                    hg.fixed_point(pert, student_params, 5, inner_opt, loss_outer) 

                    # bn_freezer.restore(self.student)

                    # FIXME ad-hoc do loss separately.
                    # self.student.train()
                    student_logits, *student_activations = self.student(x_pseudo)
                    kl_loss = self.KT_loss_student(
                        student_logits, student_activations, teacher_logits, teacher_activations)
                    kl_loss.backward()

                    self.optimizer_student.step()

                    with torch.no_grad():
                        student_total_loss = loss_outer(pert, student_params)
                else:
                    
                    student_logits, *student_activations = self.student(x_pseudo)
                    student_total_loss = self.KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations)
                    
                    # Backdoor suppression
                    if self.args.sup_backdoor and not self.args.use_hg and self.n_pseudo_batches >= self.args.hg_start_round:
                        batch_pert = torch.zeros_like(
                            x_pseudo[:1], requires_grad=True, device='cuda')
                        pert_opt = torch.optim.SGD(params=[batch_pert], lr=10)

                        for _ in range(self.args.inner_round):
                            sud_logits, *_ = self.student(x_pseudo)
                            per_logits, *_ = self.student(x_pseudo+batch_pert)

                            loss_regu = self.KT_loss_generator(per_logits, sud_logits.detach())
                            pert_opt.zero_grad()
                            loss_regu.backward(retain_graph=True)
                            pert_opt.step()
                        pert = batch_pert * min(1, 10 / torch.norm(batch_pert))

                        student_per_logits, *_ = self.student(x_pseudo + pert.detach())

                        student_total_loss = student_total_loss \
                            - 0.5 * self.KT_loss_generator(student_per_logits, student_logits)

                    self.optimizer_student.zero_grad()
                    student_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 5)
                    self.optimizer_student.step()

            if self.shuf_loss_fn is not None and self.args.pseudo_test_batches > 0:
                # to check if the shuffle model is good enough.
                if self.n_pseudo_batches >= (self.args.n_generator_iter + self.args.n_student_iter) and self.n_pseudo_batches < self.args.pseudo_test_batches:
                    self.shuf_loss_fn.add_syn_batch(x_pseudo, teacher_logits)

                if self.n_pseudo_batches + 1 > self.args.pseudo_test_batches:
                    self.shuf_loss_fn.select_shuffle(self.args.shuf_teacher, 
                        self.test_loader, self.test_poi_loader)

            ## Last call to this batch, log metrics
            if (idx_pseudo + 1) % self.n_repeat_batch == 0:
                with torch.no_grad():
                    teacher_maxes, teacher_argmaxes = torch.max(torch.softmax(teacher_logits, dim=1), dim=1)
                    student_maxes, student_argmaxes = torch.max(torch.softmax(student_logits, dim=1), dim=1)
                    running_generator_total_loss.update(float(generator_total_loss))
                    running_student_total_loss.update(float(student_total_loss))
                    running_teacher_maxes_avg.update(float(torch.mean(teacher_maxes)))
                    running_student_maxes_avg.update(float(torch.mean(student_maxes)))
                    teacher_maxes_distribution.append(teacher_maxes)
                    teacher_argmaxes_distribution.append(teacher_argmaxes)
                    student_maxes_distribution.append(student_maxes)
                    student_argmaxes_distribution.append(student_argmaxes)

                if (self.n_pseudo_batches+1) % self.args.log_freq == 0 or self.n_pseudo_batches == self.args.total_n_pseudo_batches - 1:
                    test_acc = self.test(self.student, self.test_loader)

                    with torch.no_grad():
                        log_info = {}
                        print('\nBatch {}/{} -- Generator Loss: {:02.2f} -- Student Loss: {:02.2f}'.format(self.n_pseudo_batches, self.args.total_n_pseudo_batches, running_generator_total_loss.avg(), running_student_total_loss.avg()))
                        print_info = 'Test Acc: {:02.2f}%'.format(test_acc*100)

                        test_poi_acc = self.test(self.student, self.test_poi_loader)
                        print_info += f' ASR: {test_poi_acc*100:.2f}%'
                        log_info.update({'Eval/test_poi_acc': test_poi_acc*100, })
                        
                        print_info += f" stu lr: {self.scheduler_student.get_last_lr()[0]:g}, " \
                            f"gen lr: {self.scheduler_generator.get_last_lr()[0]:g}"
                        print(print_info)
                        
                        log_info.update({
                            'TRAIN_PSEUDO/generator_total_loss': running_generator_total_loss.avg(),
                            'TRAIN_PSEUDO/student_total_loss': running_student_total_loss.avg(),
                            'TRAIN_PSEUDO/teacher_maxes_avg': running_teacher_maxes_avg.avg(),
                            'TRAIN_PSEUDO/student_maxes_avg': running_student_maxes_avg.avg(),
                            'TRAIN_PSEUDO/student_lr': self.scheduler_student.get_last_lr()[0],
                            'TRAIN_PSEUDO/generator_lr': self.scheduler_generator.get_last_lr()[0],
                            'TIME/data_time_sec': running_data_time.avg(),
                            'TIME/batch_time_sec': running_batch_time.avg(),
                            'Eval/test_acc': test_acc*100,
                            'TRAIN_SHUFFLE/shuffle_kl_loss': shuffle_kl_loss_avg.avg(),
                            'TRAIN_SHUFFLE/shuffle_kl_loss_std': shuffle_kl_loss_avg.std(),
                            'TRAIN_SHUFFLE/shuffle_mask': shuffle_mask_avg.avg(),
                            'TRAIN_SHUFFLE/shuffle_mask_std': shuffle_mask_avg.std(),
                            'TRAIN_SHUFFLE/shuffle_drop': shuffle_drop_avg.avg(),
                        })

                        wandb.log(log_info, commit=False)

                        running_data_time.reset(), running_batch_time.reset()
                        running_teacher_maxes_avg.reset(), running_student_maxes_avg.reset()
                        running_generator_total_loss.reset(), running_student_total_loss.reset(),
                        teacher_maxes_distribution, teacher_argmaxes_distribution = [], []
                        student_maxes_distribution, student_argmaxes_distribution = [], []
                        shuffle_kl_loss_avg.reset(), shuffle_mask_avg.reset(), shuffle_drop_avg.reset()
                    
                    wandb.log({'log step': (self.n_pseudo_batches+1) // self.args.log_freq}, commit=True)

                    if self.args.save_n_checkpoints > 1:
                        if (self.n_pseudo_batches+1) // self.args.log_freq > self.args.log_times - self.args.save_n_checkpoints:
                            self.save_model(log_info, f"pb{self.n_pseudo_batches}.pth")
                    self.save_model(log_info, f"last.pth")

                self.n_pseudo_batches += 1
                self.scheduler_student.step()
                self.scheduler_generator.step()

            idx_pseudo += 1
            running_batch_time.update(time() - end)
            end = time()

        # test_acc = self.test(self.student, self.test_loader)
        if self.args.save_final_model:  # make sure last epoch saved
            self.save_model(log_info)

        return test_acc*100


    def test(self, model, data_loader):

        model.eval()
        running_test_acc = AggregateScalar()

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.args.device), y.to(self.args.device)
                student_logits, *student_activations = model(x)
                acc = accuracy(student_logits.data, y, topk=(1,))[0]
                running_test_acc.update(float(acc), x.shape[0])

        # model.train()  # do not change the mode of teacher.
        return running_test_acc.avg()


    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()


    def divergence(self, student_logits, teacher_logits, reduction='mean'):
        divergence = F.kl_div(F.log_softmax(student_logits / self.args.KL_temperature, dim=1), F.softmax(teacher_logits / self.args.KL_temperature, dim=1), reduction=reduction)  # forward KL

        return divergence


    def KT_loss_generator(self, student_logits, teacher_logits, reduction='mean'):

        divergence_loss = self.divergence(student_logits, teacher_logits, reduction=reduction)
        total_loss = - divergence_loss

        return total_loss


    def KT_loss_student(self, student_logits, student_activations, teacher_logits, teacher_activations):

        divergence_loss = self.divergence(student_logits, teacher_logits)
        if self.args.AT_beta > 0:
            at_loss = 0
            for i in range(len(student_activations)):
                at_loss = at_loss + self.args.AT_beta * self.attention_diff(student_activations[i], teacher_activations[i])
        else:
            at_loss = 0

        total_loss = divergence_loss + at_loss

        return total_loss
    
    def resume_model(self, save_path=None, save_name="last.pth", ignore_student_opt=False):
        if save_path is None:
            save_path = self.save_model_path
        checkpoint_path = os.path.join(save_path, save_name)
        if os.path.isfile(checkpoint_path):
            print(f"Load checkpoint <= {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            print(' Resuming at batch iter {} with'.format(checkpoint['n_pseudo_batches']))
            print(f" Acc {checkpoint['Eval/test_acc']}")
            if 'Eval/test_poi_acc' in checkpoint:
                print(f" ASR {checkpoint['Eval/test_poi_acc']}")
            print('Running an extra {} iterations'.format(self.args.total_n_pseudo_batches - checkpoint['n_pseudo_batches']))
            self.n_pseudo_batches = checkpoint['n_pseudo_batches']
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
            self.scheduler_generator.load_state_dict(checkpoint['scheduler_generator'])
            self.student.load_state_dict(checkpoint['student_state_dict'])
            if not ignore_student_opt:
                self.optimizer_student.load_state_dict(checkpoint['optimizer_student'])
                self.scheduler_student.load_state_dict(checkpoint['scheduler_student'])
            if 'shuf_loss_fn' in checkpoint:
                self.shuf_loss_fn.load_state_dict(checkpoint['shuf_loss_fn'])
        else:
            raise FileNotFoundError(f"Not found checkpoint at {checkpoint_path}")

    def save_model(self, info, file_name=None):
        make_if_not_exist(self.save_model_path)
        save_dict = {'args': self.args,
                    'n_pseudo_batches': self.n_pseudo_batches,
                    'generator_state_dict': self.generator.state_dict(),
                    'student_state_dict': self.student.state_dict(),
                    'optimizer_generator': self.optimizer_generator.state_dict(),
                    'optimizer_student': self.optimizer_student.state_dict(),
                    'scheduler_generator': self.scheduler_generator.state_dict(),
                    'scheduler_student': self.scheduler_student.state_dict(),
                    # self.shuf_loss_fn.pseudo_dataset.
                    **info}
        
        if self.shuf_loss_fn is not None:
            save_dict['shuf_loss_fn'] = self.shuf_loss_fn.state_dict()
        if file_name is not None:
            fname = os.path.join(self.save_model_path, file_name)
            torch.save(save_dict, fname)
            print(f"Save model => {fname}")
        fname = os.path.join(self.save_model_path, 'last.pth')
        torch.save(save_dict, fname)
        print(f"Save model => {fname}")

    def save_generator(self):
        fname = os.path.join(self.save_image_path, f'generator.pth')
        torch.save({'state_dict': self.generator.state_dict()}, fname)
        print(f"Save generator to {fname}")
    
    def save_pseudo_images(self, i_iter, images, **kwargs):
        img_folder = self.get_pseudo_images_folder()
        fname = os.path.join(img_folder, f'{i_iter}.pth')
        make_if_not_exist(img_folder)
        torch.save({'imgs': images.data.cpu(), **kwargs}, fname)
        # print(f"Save images to {fname}")

    def get_pseudo_images_folder(self):
        img_folder = os.path.join(self.save_image_path, 'images')
        return img_folder
    
    