import argparse
import cv2
import numpy as np
import torch

from model import ResnetGenerator
from datasets import PairedDataset
from utils import *

# Arguments
parser = argparse.ArgumentParser(description='Test UGATIT')

parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--exp_num', type=int, default=2)

# Training parameters
parser.add_argument('--iterations', type=int, default=10 * 100000)

# Model
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--n_feats', type=int, default=64)
parser.add_argument('--n_res', type=int, default=4)

# Dataset
parser.add_argument('--domain1', type=str, default='Selfie')
parser.add_argument('--domain2', type=str, default='Anime')
parser.add_argument('--train_size', type=int, default=3400)

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--flip', type=bool, default=False)
parser.add_argument('--normalize', type=bool, default=True)

opt = parser.parse_args()


def Test_UGATIT(args):
    # Device
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Random Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Models
    genA2B = ResnetGenerator(3, 3, n_feats=args.n_feats, n_blocks=args.n_res, img_size=args.patch_size).to(device)
    genB2A = ResnetGenerator(3, 3, n_feats=args.n_feats, n_blocks=args.n_res, img_size=args.patch_size).to(device)

    checkpoint_dir = './experiments/exp{}/checkpoints/'.format(args.exp_num)
    genA2B.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'genA2B_{}steps.pth'.format(args.iterations)), map_location=device))
    genB2A.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'genB2A_{}steps.pth'.format(args.iterations)), map_location=device))

    genA2B.eval()
    genB2A.eval()

    # Transform
    transform = A.Compose(get_transforms(args), additional_targets={'target': 'image'})

    # Dataset
    test_dataset = PairedDataset(domain1=args.domain1, domain2=args.domain2, train_size=args.train_size, train=False, transform=transform)

    # Evaluate
    save_dir = './experiments/exp{}/results/{}iterations'.format(args.exp_num, args.iterations)
    domain1to2_dir = os.path.join(save_dir, '{}2{}'.format(args.domain1, args.domain2))
    domain2to1_dir = os.path.join(save_dir, '{}2{}'.format(args.domain2, args.domain1))
    if not os.path.exists(domain1to2_dir):
        os.makedirs(domain1to2_dir)
    if not os.path.exists(domain2to1_dir):
        os.makedirs(domain2to1_dir)

    for index, data in enumerate(test_dataset):
        real_A, real_B = data['domain1'], data['domain2']
        real_A, real_B = real_A.to(device), real_B.to(device)
        real_A, real_B = torch.unsqueeze(real_A, dim=0), torch.unsqueeze(real_B, dim=0)

        # real_A and its transformations
        fake_A2B, _, fake_A2B_heatmap = genA2B(real_A)
        fake_A2B2A, _, fake_A2B2A_heatmap = genB2A(fake_A2B)
        fake_A2A, _, fake_A2A_heatmap = genB2A(real_A)

        A2B = np.concatenate((
            RGB2BGR(tensor_to_numpy(denorm(real_A[0].cpu()))),
            # cam(tensor_to_numpy(fake_A2A_heatmap[0].cpu()), size=args.patch_size),
            # RGB2BGR(tensor_to_numpy(denorm(fake_A2A[0].cpu()))),
            # cam(tensor_to_numpy(fake_A2B_heatmap[0].cpu()), size=args.patch_size),
            RGB2BGR(tensor_to_numpy(denorm(fake_A2B[0].cpu()))),
            # cam(tensor_to_numpy(fake_A2B2A_heatmap[0].cpu()), size=args.patch_size),
            # RGB2BGR(tensor_to_numpy(denorm(fake_A2B2A[0].cpu())))
        ), axis=1)

        # real_B and its transformations
        fake_B2A, _, fake_B2A_heatmap = genB2A(real_B)
        fake_B2A2B, _, fakeB2A2B_heatmap = genA2B(fake_B2A)
        fake_B2B, _, fake_B2B_heatmap = genB2A(real_B)

        B2A = np.concatenate((
            RGB2BGR(tensor_to_numpy(denorm(real_B[0].cpu()))),
            # cam(tensor_to_numpy(fake_B2B_heatmap[0].cpu()), size=args.patch_size),
            # RGB2BGR(tensor_to_numpy(denorm(fake_B2B[0].cpu()))),
            # cam(tensor_to_numpy(fake_B2A_heatmap[0].cpu()), size=args.patch_size),
            RGB2BGR(tensor_to_numpy(denorm(fake_B2A[0].cpu()))),
            # cam(tensor_to_numpy(fakeB2A2B_heatmap[0].cpu()), size=args.patch_size),
            # RGB2BGR(tensor_to_numpy(denorm(fake_B2A2B[0].cpu())))
        ), axis=1)

        cv2.imwrite(os.path.join(domain1to2_dir, '{}.png'.format(index+1)), A2B)
        cv2.imwrite(os.path.join(domain2to1_dir, '{}.png'.format(index+1)), B2A)


if __name__ == "__main__":
    Test_UGATIT(args=opt)
