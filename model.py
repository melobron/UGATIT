import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


################################# Generator #################################
class ResnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=64, n_blocks=6, img_size=256):
        super(ResnetGenerator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feats = n_feats
        self.n_blocks = n_blocks

        # Encoder
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_feats, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(n_feats),
            nn.ReLU(True)
        ]

        n_down = 2
        for i in range(n_down):
            mult = 2**i
            encoder += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_feats*mult, n_feats*mult*2, 3, 2, 0, bias=False),
                nn.InstanceNorm2d(n_feats*mult*2),
                nn.ReLU(True)
            ]

        mult = 2**n_down
        for i in range(n_blocks):
            encoder += [ResnetBlock(n_feats*mult, use_bias=False)]

        self.encoder = nn.Sequential(*encoder)

        # Class Activation Map
        self.gap_fc = nn.Linear(n_feats*mult, 1, bias=False)
        self.gmp_fc = nn.Linear(n_feats*mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(n_feats*mult*2, n_feats*mult, 1, 1, 0, bias=True)
        self.relu = nn.ReLU(True)

        # FC Block (Gamma, Beta)
        fc = [
            nn.Linear(img_size // mult * img_size // mult * n_feats * mult, n_feats*mult, bias=False),
            nn.ReLU(True),
            nn.Linear(n_feats*mult, n_feats*mult, bias=False),
            nn.ReLU(True)
        ]
        self.gamma = nn.Linear(n_feats*mult, n_feats*mult, bias=False)
        self.beta = nn.Linear(n_feats*mult, n_feats*mult, bias=False)

        self.fc = nn.Sequential(*fc)

        # Decoder
        for i in range(n_blocks):
            setattr(self, 'ResnetAdaLINBlock_{}'.format(i+1), ResnetAdaLINBlock(n_feats*mult, use_bias=False))

        upsampling = []
        for i in range(n_down):
            mult = 2**(n_down - i)
            upsampling += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_feats*mult, (n_feats*mult) // 2, 3, 1, 0, bias=False),
                LIN((n_feats*mult) // 2),
                nn.ReLU(True)
            ]

        upsampling += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_feats, out_channels, 7, 1, 0, bias=False),
            nn.Tanh()
        ]

        self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Class Activation Map
        gap = F.adaptive_avg_pool2d(x, 1)  # (b, c, 1, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))  # (b, c) -> (b, 1)
        gap_weight = list(self.gap_fc.parameters())[0]  # parameter (c)
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)  # (b, c, h, w)

        gmp = F.adaptive_max_pool2d(x, 1)  # (b, c, 1, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))  # (b, c) -> (b, 1)
        gmp_weight = list(self.gmp_fc.parameters())[0]  # parameter (c)
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)  # (b, c, h, w)

        cam_logit = torch.cat([gap_logit, gmp_logit], dim=1)  # (b, 2)
        x = torch.cat([gap, gmp], dim=1)  # # (b, 2c, h, w)
        x = self.relu(self.conv1x1(x))  # # (b, c, h, w)

        heatmap = torch.sum(x, dim=1, keepdim=True)  # # (b, 1, h, w)

        # FC Block (Gamma, Beta)
        x_ = self.fc(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        # Decoder
        for i in range(self.n_blocks):
            x = getattr(self, 'ResnetAdaLINBlock_{}'.format(i+1))(x, gamma, beta)
        out = self.upsampling(x)

        return out, cam_logit, heatmap


################################# Generator Blocks #################################
class ResnetBlock(nn.Module):
    def __init__(self, n_feats, use_bias):
        super(ResnetBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 0, bias=use_bias),
            nn.InstanceNorm2d(n_feats),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 0, bias=use_bias),
            nn.InstanceNorm2d(n_feats),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = x + self.model(x)
        return out


class ResnetAdaLINBlock(nn.Module):
    def __init__(self, n_feats, use_bias):
        super(ResnetAdaLINBlock, self).__init__()

        self.layer1 = nn.Sequential(*[
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 0, bias=use_bias),
        ])

        self.adalin1 = AdaLIN(n_feats)
        self.relu = nn.ReLU(True)

        self.layer2 = nn.Sequential(*[
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_feats, n_feats, 3, 1, 0, bias=use_bias),
        ])

        self.adalin2 = AdaLIN(n_feats)

    def forward(self, x, gamma, beta):
        out = self.layer1(x)
        out = self.adalin1(out, gamma, beta)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.adalin2(out, gamma, beta)
        return out + x


class AdaLIN(nn.Module):
    def __init__(self, n_feats, eps=1e-5):
        super(AdaLIN, self).__init__()

        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, n_feats, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class LIN(nn.Module):
    def __init__(self, n_feats, eps=1e-5):
        super(LIN, self).__init__()

        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, n_feats, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, n_feats, 1, 1))
        self.beta = Parameter(torch.Tensor(1, n_feats, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + self.beta.expand(x.shape[0], -1, -1, -1)
        return out


################################# Discriminator #################################
class Discriminator(nn.Module):
    def __init__(self, in_channels, n_feats=64, n_layers=5):
        super(Discriminator, self).__init__()

        # Encoder
        encoder = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channels, n_feats, 4, 2, 0, bias=True)),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(1, n_layers-2):
            mult = 2**(i-1)
            encoder += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(nn.Conv2d(n_feats*mult, n_feats*mult*2, 4, 2, 0, bias=True)),
                nn.LeakyReLU(0.2, True)
            ]

        mult = 2**(n_layers-3)
        encoder += [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(n_feats*mult, n_feats*mult*2, 4, 1, 0, bias=True)),
            nn.LeakyReLU(0.2, True)
        ]

        self.encoder = nn.Sequential(*encoder)

        # Class Activation Map
        mult = 2**(n_layers-2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(n_feats*mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(n_feats*mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(n_feats*mult*2, n_feats*mult, 1, 1, 0, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        # Classifier
        classifier = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(n_feats*mult, 1, 4, 1, 0, bias=False))
        ]

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Class Activation Map
        gap = F.adaptive_avg_pool2d(x, 1)  # (b, c, 1, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))  # (b, c) -> (b, 1)
        gap_weight = list(self.gap_fc.parameters())[0]  # parameter (c)
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)  # (b, c, h, w)

        gmp = F.adaptive_max_pool2d(x, 1)  # (b, c, 1, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))  # (b, c) -> (b, 1)
        gmp_weight = list(self.gmp_fc.parameters())[0]  # parameter (c)
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)  # (b, c, h, w)

        cam_logit = torch.cat([gap_logit, gmp_logit], dim=1)  # (b, 2)
        x = torch.cat([gap, gmp], dim=1)  # # (b, 2c, h, w)
        x = self.leaky_relu(self.conv1x1(x))  # # (b, c, h, w)

        heatmap = torch.sum(x, dim=1, keepdim=True)  # # (b, 1, h, w)

        out = self.classifier(x)
        return out, cam_logit, heatmap


################################# ETC #################################
class RhoClipper:
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


################################# main #################################
if __name__ == "__main__":
    generator = ResnetGenerator(in_channels=3, out_channels=3, n_feats=64, n_blocks=6, img_size=256)
    discriminator = Discriminator(in_channels=3, n_feats=64, n_layers=5)

    input = torch.randn(10, 3, 256, 256)
    print('input:{}'.format(input.shape))

    output1, cam1, heatmap1 = generator(input)
    print('output1:{}'.format(output1.shape))
    print('cam1:{}'.format(cam1.shape))
    print('heatmap1:{}'.format(heatmap1.shape))

    output2, cam2, heatmap2 = discriminator(output1)
    print('output2:{}'.format(output2.shape))
    print('cam2:{}'.format(cam2.shape))
    print('heatmap2:{}'.format(heatmap2.shape))
