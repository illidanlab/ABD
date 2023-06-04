import torch
from torch import nn
import random
import numpy as np
import utils.hypergrad as hg

class UnlearnOptimizer(nn.Module):
    def __init__(self, kl_div, inner_round=10) -> None:
        super().__init__()
        self.kl_div = kl_div
        self.pert_lr = 0.1
        self.inner_round = inner_round
        self.hg_out_portion = 0.02

    def step(self, student, teacher, optimizer_student, x_pseudo, criterion):
        with torch.no_grad():
            teacher_logits = teacher(x_pseudo)

        # generate noise
        batch_pert = torch.zeros_like(
            x_pseudo[:1], requires_grad=True, device='cuda')
        pert_opt = torch.optim.Adam(params=[batch_pert], lr=self.pert_lr)
        pert_model = teacher

        pert_model.eval()
        for i_pert in range(self.inner_round):
            with torch.no_grad():
                sud_logits = pert_model(x_pseudo)
            per_logits = pert_model(x_pseudo+batch_pert)

            # loss_regu = torch.mean(cos(sud_logits.detach(), per_logits))
            pert_loss = - torch.mean(self.kl_div(per_logits, sud_logits.detach()))
            # print(f" [{i_pert}] pert loss: {pert_loss.item():.4f}")
            pert_opt.zero_grad()
            pert_loss.backward(retain_graph=True)
            pert_opt.step()
        pert = batch_pert  # TODO may not be useful: * min(1, 10 / torch.norm(batch_pert))

        # hg losses
        #define the inner loss L2
        def loss_inner(perturb, model_params):
            images = x_pseudo.cuda()
            per_img = images+perturb[0]
            
            sud_logits = student(images)
            sud_logits = sud_logits.detach()
            per_logits = student(per_img)
            
            pert_loss = torch.mean(self.kl_div(per_logits, sud_logits))
            pert_loss = - pert_loss +0.001*torch.pow(torch.norm(perturb[0]),2)
            return pert_loss

        def loss_outer(perturb, model_params):
            portion = self.hg_out_portion # 0.02 # increase this to increase batch size.
            patching = torch.zeros_like(x_pseudo, device='cuda')
            number = x_pseudo.shape[0]
            rand_idx = random.sample(list(np.arange(number)),int(number*portion))
            patching[rand_idx] = perturb[0]
            unlearn_imgs = x_pseudo+patching

            student_logits = student(x_pseudo)
            student_per_logits = student(unlearn_imgs)

            pert_loss = torch.mean(self.kl_div(student_per_logits, student_logits))

            loss = pert_loss
            
            return loss

        # optimize
        optimizer_student.zero_grad()

        inner_opt = hg.GradientDescent(loss_inner, 0.1)
        hg.fixed_point(pert, list(student.parameters()), 5, inner_opt, loss_outer) 

        student_logits = student(x_pseudo)  # This may increase the memory a lot.
        kl_loss = criterion(student_logits, teacher_logits.detach())
        kl_loss.backward()

        optimizer_student.step()

        with torch.no_grad():
            student_logits = student(x_pseudo)
        return student_logits, teacher_logits
