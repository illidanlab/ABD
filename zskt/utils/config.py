"""Configuration file for defining paths to data."""
import os


def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)


hostname = os.uname()[1]  # type: str
# Update your paths here.
CHECKPOINT_ROOT = './chkpts'
if hostname.startswith('illidan') and int(hostname.split('-')[-1]) >= 8:
    data_root = '/localscratch2/jyhong/'
elif hostname.startswith('illidan'):
    data_root = '/media/Research/jyhong/data'
else:  # other servers
    # TODO set up root to data.
    data_root = os.path.expanduser('~/data')
make_if_not_exist(data_root)
make_if_not_exist(CHECKPOINT_ROOT)

DATA_PATHS = {}
if hostname.startswith('illidan') and int(hostname.split('-')[-1]) < 8:  # for illidan lab servers.
    # personal config
    home_path = os.path.expanduser('~/')

    if int(hostname.split('-')[-1]) == 7:
        BDBlocker_path = '/home/jyhong/projects_ex/BackdoorBlocker/'
    else:
        BDBlocker_path = '/home/jyhong/projects/BackdoorBlocker/'
else:  # all other servers.
    # TODO specify your path root to pretrained models.
    BDBlocker_path = os.path.expanduser('~/projects/BackdoorBlocker/')

backdoor_PT_model_path = BDBlocker_path + 'results/normal_training/'

DATA_PATHS["CIFAR10"] = data_root + '/Cifar10'
DATA_PATHS["GTSRB"] = data_root + '/gtsrb'
DATA_PATHS["one-image"] = data_root + '/one-image'
DATA_PATHS["pubfig"] = data_root + '/pubfig'
