import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import itertools
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from model import Generator, Discriminator, RhoClipper
from datasets import TranslationDataset
from utils import *


class TrainUGATIT:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.offset_epochs = args.offset_epochs
        self.decay_epochs = args.decay_epochs
        self.checkpoint_interval = args.checkpoint_interval

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.mean = args.mean
        self.std = args.std

        self.domain1 = args.domain1
        self.domain2 = args.domain2
        self.domain1_size = args.domain1_size
        self.domain2_size = args.domain2_size
        self.test_size = args.test_size

        self.n_feats = args.n_feats
        self.n_res = args.n_res
        self.patch_size = args.patch_size

        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        # Models
        self.genA2B = Generator(n_feats=self.n_feats, n_blocks=self.n_res, img_size=self.patch_size).to(self.device)
        self.genB2A = Generator(n_feats=self.n_feats, n_blocks=self.n_res, img_size=self.patch_size).to(self.device)
        self.disGA = Discriminator(n_feats=self.n_feats, n_layers=7).to(self.device)
        self.disGB = Discriminator(n_feats=self.n_feats, n_layers=7).to(self.device)
        self.disLA = Discriminator(n_feats=self.n_feats, n_layers=5).to(self.device)
        self.disLB = Discriminator(n_feats=self.n_feats, n_layers=5).to(self.device)

        # Loss
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # Optimizer
        self.optimizer_G = optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                      lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.optimizer_D = optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                                                      self.disLA.parameters(), self.disLB.parameters()),
                                      lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        # Scheduler
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)
        self.scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=LambdaLR(
            args.n_epochs, args.offset_epochs, args.decay_epochs).step)

        # Rho Clipper
        self.Rho_clipper = RhoClipper(0, 1)

        # Transform
        train_transform = transforms.Compose(get_transforms(args))
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Dataset
        self.train_dataset = TranslationDataset(domain1=self.domain1, domain2=self.domain2,
                                                domain1_size=self.domain1_size, domain2_size=self.domain2_size,
                                                train=True, transform=train_transform)
        # self.test_dataset = TranslationDataset(domain1=self.domain1, domain2=self.domain2,
        #                                         domain1_size=self.domain1_size, domain2_size=self.domain2_size,
        #                                         train=False, transform=test_transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(self.device)
        self.prepare()

        for epoch in range(1, self.n_epochs + 1):
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                ########### Training ###########
                self.genA2B.train(), self.genB2A.train()
                self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                for batch, data in enumerate(tepoch):
                    real_A, real_B = data['domain1'], data['domain2']
                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    ########### Update D ###########
                    self.optimizer_D.zero_grad()

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
                    self.optimizer_D.step()

                    ########### Update G ###########
                    self.optimizer_G.zero_grad()

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
                    self.optimizer_G.step()

                    # Rho Clipper
                    self.genA2B.apply(self.Rho_clipper)
                    self.genB2A.apply(self.Rho_clipper)

                    ########### Save Results ###########
                    tepoch.set_postfix(Total_loss=Generator_loss.item() + Discriminator_loss.item(),
                                       G_loss=Generator_loss.item(), D_loss=Discriminator_loss.item())
                    self.summary.add_scalar('Total_loss', Generator_loss.item() + Discriminator_loss.item(), epoch)
                    self.summary.add_scalar('G_loss', Generator_loss.item(), epoch)
                    self.summary.add_scalar('D_loss', Discriminator_loss.item(), epoch)

                ########### Validation ###########



            self.scheduler_G.step()
            self.scheduler_D.step()

            # Checkpoints
            if epoch % self.checkpoint_interval == 0 or epoch == self.n_epochs:
                torch.save(self.genA2B.state_dict(), os.path.join(self.checkpoint_dir, 'netG_A2B_{}epochs.pth'.format(epoch)))
                torch.save(self.genB2A.state_dict(), os.path.join(self.checkpoint_dir, 'netG_B2A_{}epochs.pth'.format(epoch)))

        self.summary.close()
