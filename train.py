import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import time
from torch.utils.tensorboard import SummaryWriter
import json

from model import ResnetGenerator, Discriminator, RhoClipper
from dataset import PairedDataset
from utils import *


class TrainUGATIT:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(args.seed)

        # Training Parameters
        self.iterations = args.iterations
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.n_feats = args.n_feats
        self.n_res = args.n_res

        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        # Models
        self.genA2B = ResnetGenerator(in_channels=3, out_channels=3, n_feats=self.n_feats, n_blocks=self.n_res,
                                      img_size=self.patch_size).to(self.device)
        self.genB2A = ResnetGenerator(in_channels=3, out_channels=3, n_feats=self.n_feats, n_blocks=self.n_res,
                                      img_size=self.patch_size).to(self.device)
        self.disGA = Discriminator(in_channels=3, n_feats=self.n_feats, n_layers=7).to(self.device)
        self.disGB = Discriminator(in_channels=3, n_feats=self.n_feats, n_layers=7).to(self.device)
        self.disLA = Discriminator(in_channels=3, n_feats=self.n_feats, n_layers=5).to(self.device)
        self.disLB = Discriminator(in_channels=3, n_feats=self.n_feats, n_layers=5).to(self.device)

        # Loss
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # Optimizer
        self.G_optim = optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr,
                                  betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        # Rho Clipper
        self.Rho_clipper = RhoClipper(0, 1)

        # Transform
        transform = A.Compose(get_transforms(args), additional_targets={'target': 'image'})

        # Dataset
        self.dataset = PairedDataset(domain1=args.domain1, domain2=args.domain2, train_size=args.train_size, train=True,
                                     transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)

        # Save Paths
        self.exp_dir, self.exp_num = make_exp_dir('./experiments/')['new_dir'], make_exp_dir('./experiments/')[
            'new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.result_dir = os.path.join(self.exp_dir, 'results')
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(args.__dict__, f, indent=4)

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def train(self):
        print(self.device)

        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start = time.time()
        for step in range(1, self.iterations + 1):
            if step > (self.iterations // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iterations // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iterations // 2))

            try:
                real_A, real_B = train_iter.next()['domain1'], train_iter.next()['domain2']
            except:
                train_iter = iter(self.dataloader)
                real_A, real_B = train_iter.next()['domain1'], train_iter.next()['domain2']

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update Discriminator
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) \
                           + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) \
                               + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) \
                           + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) \
                               + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))

            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) \
                           + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) \
                               + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) \
                           + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) \
                               + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update Generator
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fakeA2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fakeB2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fakeA2A_cam_logit, torch.zeros_like(fakeA2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fakeB2B_cam_logit, torch.zeros_like(fakeB2B_cam_logit).to(self.device))

            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) \
                       + self.cycle_weight * G_recon_loss_A \
                       + self.identity_weight * G_identity_loss_A \
                       + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) \
                       + self.cycle_weight * G_recon_loss_B \
                       + self.identity_weight * G_identity_loss_B \
                       + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # Rho Clipper
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            print('[{}/{}] time:{} d_loss:{} g_loss:{}'.format(
                step, self.iterations, time.time() - start, Discriminator_loss, Generator_loss))

            if step % 1000 == 0:
                # Visualize Losses
                self.summary.add_scalars('D_Adv_losses', {
                    'GA':D_ad_loss_GA.item(), 'GB':D_ad_loss_GB.item(), 'LA':D_ad_loss_LA.item(), 'LB':D_ad_loss_LB.item(),
                    'GA_CAM':D_ad_cam_loss_GA.item(), 'GB_CAM':D_ad_cam_loss_GB.item(),
                    'LA_CAM':D_ad_cam_loss_LA.item(), 'LB_CAM':D_ad_cam_loss_LB.item()
                }, step)
                self.summary.add_scalars('G_Adv_losses', {
                    'GA':G_ad_loss_GA.item(), 'GB':G_ad_loss_GB.item(), 'LA':G_ad_loss_LA.item(), 'LB':G_ad_loss_LB.item(),
                    'GA_CAM':G_ad_cam_loss_GA.item(), 'GB_CAM':G_ad_cam_loss_GB.item(),
                    'LA_CAM':G_ad_cam_loss_LA.item(), 'LB_CAM':G_ad_cam_loss_LB.item()
                }, step)
                self.summary.add_scalars('G_losses', {
                    'recon_A':G_recon_loss_A.item(), 'recon_B':G_recon_loss_B.item(),
                    'identity_A':G_identity_loss_A.item(),'identity_B':G_identity_loss_B.item(),
                    'CAM_A':G_cam_loss_A.item(), 'CAM_B':G_cam_loss_B.item()
                }, step)
                self.summary.add_scalars('Total_losses', {
                    'G_total_loss': Generator_loss, 'D_total_loss': Discriminator_loss
                }, step)

            # Save weights
            if step % 100000 == 0:
                torch.save(self.genA2B.state_dict(), os.path.join(self.checkpoint_dir, 'genA2B_{}steps.pth'.format(step)))
                torch.save(self.genB2A.state_dict(), os.path.join(self.checkpoint_dir, 'genB2A_{}steps.pth'.format(step)))

        self.summary.close()
