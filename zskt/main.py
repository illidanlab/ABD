import torch
import argparse
from solver import ZeroShotKTSolver
from utils.utils import set_torch_seeds, str2bool
import wandb
from utils.config import CHECKPOINT_ROOT


def main(args):
    """
    Run the experiment as many times as there
    are seeds given, and write the mean and std
    to as an empty file's name for cleaner logging
    """

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    set_torch_seeds(args.seed)

    solver = ZeroShotKTSolver(args)
    test_acc = solver.run()
    print('\n\nFINAL TEST ACC RATE: {:02.2f}'.format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ABD-ZSKT')

    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'GTSRB', 'pubfig'])
    parser.add_argument('--total_n_pseudo_batches', type=float, default=8e4)
    parser.add_argument('--log_times', type=int, default=400, help='total times of logging.')
    parser.add_argument('--n_generator_iter', type=int, default=1, help='per batch, for few and zero shot')
    parser.add_argument('--n_student_iter', type=int, default=10, help='per batch, for few and zero shot')
    parser.add_argument('--batch_size', type=int, default=128, help='for few and zero shot')
    parser.add_argument('--z_dim', type=int, default=100, help='for few and zero shot')
    parser.add_argument('--student_learning_rate', type=float, default=2e-3)
    parser.add_argument('--generator_learning_rate', type=float, default=1e-3)
    parser.add_argument('--teacher_architecture', type=str, default='WRN-16-2')
    parser.add_argument('--student_architecture', type=str, default='WRN-16-1')
    parser.add_argument('--KL_temperature', type=float, default=1, help='>1 to smooth probabilities in divergence loss, or <1 to sharpen them')
    parser.add_argument('--AT_beta', type=float, default=250, help='beta coefficient for AT loss')

    parser.add_argument('--pretrained_models_path', type=str, default='target0-ratio0.1_e200-b128-sgd-lr0.1-wd0.0005-cos-holdout0.05')
    parser.add_argument('--sel_model', type=str, default='best_clean_acc',
                        choices=['best_clean_acc', 'latest'], help='select teacher model.')
    parser.add_argument('--save_final_model', type=str2bool, default=True)
    parser.add_argument('--save_n_checkpoints', type=int, default=0)
    parser.add_argument('--save_model_path', type=str, default=CHECKPOINT_ROOT)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    # backdoor
    parser.add_argument('--trigger_pattern', type=str, default=None, help='refer to Haotao backdoor codes.')
    parser.add_argument('--poi_target', type=int, default=0,
                        help='target class by backdoor. Should be the same as training.')
    
    # backdoor defense
    # 1. robust distill
    parser.add_argument('--sup_backdoor', type=str2bool, default=False, help='suppress backdoor.')
    parser.add_argument('--use_hg', type=str2bool, default=False, help='Use hyper-grad for sup_backdoor.')
    parser.add_argument('--hg_start_round', type=int, default=0)
    parser.add_argument('--inner_round', type=int, default=1, help='Rounds of trigger synthesis.')
    parser.add_argument('--hg_out_portion', type=float, default=0.02)
    parser.add_argument('--sup_resume', type=str, default=None, help='which file to resume. Use pb<i_batch>.pth or last.pth')
    parser.add_argument('--sup_resume_lr', type=float, default=5e-4,
                        help='lr for sup after resume.')
    parser.add_argument('--sup_pert_model', type=str, default='student', choices=['teacher', 'student'], help='use which model to generate noise.')
    parser.add_argument('--pert_lr', default=0.1, type=float, help='lr for generate pert.')
    # 2. shuffle filter
    parser.add_argument('--shuf_teacher', type=float, default=0., help='test shuffled teacher.')
    parser.add_argument('--n_shuf_ens', type=int, default=3, help='num of ensembled shuffle models.')
    parser.add_argument('--n_shuf_layer', type=int, default=5, help='num of shuffled layers. Count from the output conv layer. Use negative to count from the input layer.')
    parser.add_argument('--neg_drop_loss', type=float, default=0., help='coef for the loss on dropped samples')
    
    parser.add_argument('--no_shuf_grad', action='store_true', help='disable grad through shuffle teacher.')
    parser.add_argument('--pseudo_test_batches', default=0,  # 50,
                        type=int, help='Set non-zero value to do vaccine verification.')
    args = parser.parse_args()

    args.norm_inp = True  # normalize inputs into normal range.
    args.total_n_pseudo_batches = int(args.total_n_pseudo_batches)
    if args.AT_beta > 0: assert args.student_architecture[:3] in args.teacher_architecture
    args.log_freq = max(1, int(args.total_n_pseudo_batches / args.log_times))
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args.experiment_name = f'{args.dataset}_{args.teacher_architecture}_{args.student_architecture}_s{args.seed}'
    # args.experiment_name = 'ZSKT_{}_{}_{}_gi{}_si{}_zd{}_plr{}_slr{}_bs{}_T{}_beta{}_s{}'.format(
    #     args.dataset, args.teacher_architecture,  args.student_architecture, args.n_generator_iter, args.n_student_iter, args.z_dim, args.generator_learning_rate, args.student_learning_rate, args.batch_size, args.KL_temperature, args.AT_beta, args.seed)
    if args.trigger_pattern is not None:
        args.experiment_name += f'_{args.trigger_pattern}_t{args.poi_target}'
    if args.shuf_teacher:
        args.experiment_name += f'_shtea_e{args.n_shuf_ens}_l{args.n_shuf_layer}'
        if args.neg_drop_loss > 0.:
            args.experiment_name += f'_ndl{args.neg_drop_loss}'
        if args.no_shuf_grad:
            args.experiment_name += f'_nsg'
        if args.pseudo_test_batches > 0:
            args.experiment_name += f"_ptb{args.pseudo_test_batches}"
            args.min_pseudo_test_batches = (
                args.n_generator_iter + args.n_student_iter)
            assert args.pseudo_test_batches >= args.min_pseudo_test_batches, \
                f"pseudo_test_batches should be at least {args.n_generator_iter + args.n_student_iter}"
    if args.sup_backdoor:
        if args.sup_resume is not None:
            args.exp_name_no_sb = args.experiment_name
            print(f"> exp_name_no_sb: {args.exp_name_no_sb}")
        args.experiment_name += '_sbd'
        if args.use_hg:
            args.experiment_name += '_hg'
        if args.inner_round > 1:
            args.experiment_name += f'_{args.inner_round}'
        if args.hg_out_portion != 0.:
            args.experiment_name += f'_out_hg{args.hg_out_portion}'
        if args.hg_start_round != 0:
            args.experiment_name += f'_st{args.hg_start_round}'
        if args.sup_pert_model != 'student':
            args.experiment_name += f'_pert-{args.sup_pert_model[0]}'
        if args.pert_lr != 0.1:
            args.experiment_name += f'-lr{args.pert_lr}'
    
    print(f"> experiment_name: {args.experiment_name}")
    
    if args.norm_inp and args.trigger_pattern is not None:
        # print("WARN: force to use normalized inp.")
        args.pretrained_models_path += '-ni1'

    wandb.init(project='ABD-ZSKT', name=args.experiment_name,
               config=vars(args), mode='offline' if args.no_log else 'online')

    print('\nTotal data batches: {}'.format(args.total_n_pseudo_batches))
    print('Logging results every {} batch'.format(args.log_freq))
    print('\nRunning on device: {}'.format(args.device))

    main(args)

