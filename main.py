import argparse
from train import TrainUGATIT

# Arguments
parser = argparse.ArgumentParser(description='Train UGATIT')

parser.add_argument('--exp_detail', default='Train UGATIT', type=str)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--offset_epochs', type=int, default=0)
parser.add_argument('--decay_epochs', type=int, default=100)
parser.add_argument('--checkpoint_interval', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--adv_weight', type=float, default=1.0)
parser.add_argument('--cycle_weight', type=float, default=10.0)
parser.add_argument('--identity_weight', type=float, default=10.0)
parser.add_argument('--cam_weight', type=float, default=1000.0)

# Model
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_res', type=int, default=4)

# Dataset
parser.add_argument('--domain1', type=str, default='FFHQ')
parser.add_argument('--domain2', type=str, default='Dog')
parser.add_argument('--domain1_size', type=int, default=500)
parser.add_argument('--domain2_size', type=int, default=500)
parser.add_argument('--test_size', type=int, default=100)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

args = parser.parse_args()

train_UGATIT = TrainUGATIT(args)
train_UGATIT.train()
