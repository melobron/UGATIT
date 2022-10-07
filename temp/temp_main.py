import argparse
from train import TrainUGATIT

# Arguments
parser = argparse.ArgumentParser(description='Train UGATIT')

parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)

# Training parameters
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)

# Weighted Loss
parser.add_argument('--adv_weight', type=float, default=1.0)
parser.add_argument('--cycle_weight', type=float, default=10.0)
parser.add_argument('--identity_weight', type=float, default=10.0)
parser.add_argument('--cam_weight', type=float, default=1000.0)

# Model
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_res', type=int, default=4)

# Dataset
parser.add_argument('--train_size', type=int, default=3400)
parser.add_argument('--domain1', type=str, default='Anime')
parser.add_argument('--domain2', type=str, default='Selfie')

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)

args = parser.parse_args()

train_UGATIT = TrainUGATIT(args)
train_UGATIT.train()
