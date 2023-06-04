"""Synthesis vaccine (SV) in ABD."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
from typing import Union
from utils.utils import EnsembleNet, sigmoid, PseudoDataset, PseudoDatapool


def shuffle_ckpt_layer(model, n_layers, is_alexnet=False, inplace=False):
    if not inplace:
        model = deepcopy(model)

    model_state = model.state_dict()

    total_num_layer = 0
    for k, v in model_state.items():
        if 'conv' in k or (is_alexnet and len(v.shape) == 4):
            total_num_layer += 1
    if n_layers > 0:
        shuffle_index = [0]*(total_num_layer-n_layers) + [1] * n_layers
    elif n_layers < 0:
        shuffle_index = [1] * abs(n_layers) + [0] * \
            (total_num_layer-abs(n_layers))
    else:
        return model

    new_ckpt = {}
    i = 0
    for k, v in model_state.items():
        if 'conv' in k or (is_alexnet and len(v.shape) == 4):
            if shuffle_index[i] == 1:
                _, channels, _, _ = v.size()

                idx = torch.randperm(channels)
                v = v[:, idx, ...]
            i += 1
        new_ckpt[k] = v
    model_state.update(new_ckpt)
    model.load_state_dict(model_state, strict=True)
    return model

class BackdoorSuspectLoss(nn.Module):
    def __init__(self, model, coef=1., n_shuf_ens=3, n_shuf_layer=5, no_shuf_grad=False,
                 device='cuda', batch_size=128):
        super().__init__()
        self.model = model

        self.coef = coef
        self.n_shuf_ens = n_shuf_ens
        self.n_shuf_layer = n_shuf_layer
        self.no_shuf_grad = no_shuf_grad
        self.device = device

        self.shufl_model = self.make_shuffle_suspect(model)

        self.n_shufl_penalized = 0

        self.pseudo_dataset = PseudoDatapool()
        self.pseudo_loader = DataLoader(self.pseudo_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        self.pseudo_test_flag = False

    def make_shuffle_suspect(self, model):
        print(f"Make shuffled suspect model.")
        shuf_models = []
        # NOTE ensemble shuffled models to get a stable prediction. Otherwise, the prediction highly depends on seeds.
        for _ in range(self.n_shuf_ens):
            t = shuffle_ckpt_layer(
                model, self.n_shuf_layer)
            shuf_models.append(t)
        shuffle_model = EnsembleNet(shuf_models)
        return shuffle_model

    def remake_shuffle_suspect(self, shuffle_model: EnsembleNet):
        """Re-make shuffle model inplace."""
        assert isinstance(shuffle_model, EnsembleNet)
        print(f"Re-Make shuffled suspect model.")
        # NOTE ensemble shuffled models to get a stable prediction. Otherwise, the prediction highly depends on seeds.
        for i in range(self.n_shuf_ens):
            shuffle_ckpt_layer(
                shuffle_model.models[i], self.n_shuf_layer, inplace=True)

    def test_suspect_model(self, test_loader, test_poi_loader):
        poi_kl_loss, cl_kl_loss = self.compare_shuffle(
            self.model, self.shufl_model, test_loader, test_poi_loader)
        return poi_kl_loss, cl_kl_loss

    def test_suspect_model_syn(self, pseudo_loader):
        """Test with synthetic data."""
        print(f"Test shuffle model on {len(pseudo_loader.dataset)} syn data")
        self.model.eval()
        poi_kl_loss = self.compute_divergences(
            self.model, self.shufl_model, pseudo_loader)
        poi_kl_loss = np.array(poi_kl_loss)
        poi_kl_loss = np.log10(poi_kl_loss)
        n_tail = len(poi_kl_loss[poi_kl_loss < np.mean(
            poi_kl_loss)-3*np.std(poi_kl_loss)])
        tail_ratio = n_tail / len(poi_kl_loss)
        print(f" syn data log KL loss: mean {np.mean(poi_kl_loss):.4f}, "
              f"std {np.std(poi_kl_loss):.4f}, "
              f"tail_ratio: {tail_ratio:.4f} ({n_tail}/{len(poi_kl_loss)})")
        return tail_ratio

    def loss(self, logits, x_pseudo) -> Union[float, torch.Tensor]:
        if self.coef == 0.:
            return 0.
        self.shufl_model.eval()
        if self.no_shuf_grad:
            with torch.no_grad():
                shufl_model_logits, *_ = self.shufl_model(x_pseudo)
        else:
            shufl_model_logits, *_ = self.shufl_model(x_pseudo)
        
        shufl_mask = shufl_model_logits.max(1)[1].eq(logits.max(1)[1]).float()
        # print(f"### penalized {shufl_mask.mean().item():.3f} samples.")   #  {shufl_mask.shape}")
        if shufl_mask.sum() > 0.:
            shufl_kl_loss = self.divergence(
                shufl_model_logits, logits, reduction='none')
            shufl_kl_loss = shufl_kl_loss.mean(1)  # .sum(1)
            # print(f"## shufl_kl_loss {torch.quantile(shufl_kl_loss, 0.8).item():.3f}, {shufl_mask.sum().item()} nz")
            # print(f"## shufl_kl_loss len: {shufl_kl_loss.shape}")
            shufl_kl_loss = torch.sum(
                shufl_kl_loss * shufl_mask)/shufl_mask.sum()
            # shuffle_kl_loss_avg.update(shufl_kl_loss.item())
            # shuffle_mask_avg.update(shufl_mask.sum().item())
            # print(f" ## shufl_kl_loss {shufl_kl_loss.item()}")
            # generator_total_loss = generator_total_loss + \
            #     self.args.shuf_teacher * shufl_kl_loss
            self.n_shufl_penalized = shufl_mask.sum().item()
        else:
            shufl_kl_loss = 0.
            self.n_shufl_penalized = 0
        return - self.coef * shufl_kl_loss

    def compute_divergences(self, model, shuffle_model, loader) -> list:
        kl_loss = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, *_ = model(x)
                sh_logits, *_ = shuffle_model(x)
                divergence_loss = self.divergence(
                    sh_logits, logits, reduction='none')
                # divergence_loss = divergence_loss.sum(1).data.cpu().numpy()
                divergence_loss = divergence_loss.mean(1).data.cpu().numpy()
                # kl_loss.update(divergence_loss.item(), len(x))
                kl_loss.extend(divergence_loss.tolist())
        return kl_loss

    def compare_shuffle(
            self, model, shuffle_model, test_loader, test_poi_loader):
        model.eval()
        poi_kl_loss = self.compute_divergences(
            model, shuffle_model, test_poi_loader)
        print(f" Poison KL Loss: {np.mean(poi_kl_loss):.3f}")

        cln_kl_loss = self.compute_divergences(
            model, shuffle_model, test_loader)
        print(f" Clean KL Loss: {np.mean(cln_kl_loss):.3f}")

        all_loss = np.array(poi_kl_loss + cln_kl_loss)
        all_scores = sigmoid(all_loss - np.mean(all_loss))
        all_labels = [1] * len(poi_kl_loss) + [0] * len(cln_kl_loss)

        acc = accuracy_score(all_labels, all_scores < 0.5)
        auc = roc_auc_score(all_labels, all_scores)
        print(
            f" Poi acc: {acc*100:.1f}% | auc: {auc*100:.1f}% | std: {np.std(all_loss):.3f}")

        # model.train()  # do not change the mode of teacher.
        return np.mean(poi_kl_loss), np.mean(cln_kl_loss)

    # def divergence(self, student_logits, teacher_logits, reduction='batchmean'):
    def divergence(self, student_logits, teacher_logits, reduction='mean'):
        divergence = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(
            teacher_logits, dim=1), reduction=reduction)
        return divergence

    def add_syn_batch(self, x_pseudo, teacher_logits):
        self.pseudo_dataset.add_batch(x_pseudo, teacher_logits)

    def select_shuffle(self, coef, test_loader, test_poi_loader):
        if self.pseudo_test_flag:
            return self.coef != 0.
        print("Evaluate the quality of shuffle model.")
        tail_ratio = 0.
        for i_trial in range(8):
            if i_trial > 0:
                print(f"[{i_trial}] Low tail_ratio. Re-make shuffle model...")
                self.remake_shuffle_suspect(self.shufl_model)
                self.test_suspect_model(test_loader, test_poi_loader)
            tail_ratio = self.test_suspect_model_syn(self.pseudo_loader)
            if tail_ratio > 0.02:
                print(f"Successfully found a shuffle model.")
                self.coef = coef
                success = True
                break
        else:
            self.coef = 0.
            print(f"Fail to find shuffle model! Stop")
            success = False
        self.pseudo_test_flag = True  # do not test again
        return success

    def get_extra_state(self):
        return {'pseudo_test_flag': self.pseudo_test_flag, 'coef': self.coef}
    
    def set_extra_state(self, state):
        self.pseudo_test_flag = state['pseudo_test_flag']
        self.coef = state['coef']
