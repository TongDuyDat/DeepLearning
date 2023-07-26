from math import ceil
from torch import nn
import torch
# CNNBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups = 1, act = True , bias = False):
        super(ConvBlock, self).__init__()
        padding = kernel_size//2 # 2 -> pading = 1,
        self.conv2d = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size= kernel_size, stride= stride, groups=groups, bias=bias, padding = padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.silu = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        f = self.conv2d(x)
        f = self.bn(f)
        return self.silu(f)
#Se block
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, r = 24):
        super(SqueezeExcitation,self).__init__()
        C = in_channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channel, C//r, 1)
        self.fc2 = nn.Conv2d(C//r, in_channel, 1)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        f = self.pool(x)
        f = self.fc1(f)
        f = self.silu(f)
        f = self.fc2(f)
        return x * self.sigmoid(f)

class StochasticDepth(nn.Module):
    def __init__(self, surival_pob =0.8):
        super(StochasticDepth, self).__init__()
        self.surival_pob = surival_pob
    def stochastic_depth(self, x, p = 0.8, mode = "row"):
      if p < 0 and p > 1:
        return x
      if not self.training or p == 0.0:
        return x
      pob = 1.0 - p
      if mode == "row":
        size = [x.shape[0]]+[1]*(x.ndim -1)
      else:
        size = [1]*(x.ndim -1)
      noise = torch.empty(size, dtype= x.dtype, device = x.device)
      noise = noise.bernoulli_(pob)
      if pob > 0.0:
        noise.div_(pob)
      return x*noise
    def forward(self, x):
        return self.stochastic_depth(x, p = self.surival_pob)
# Mobile Net
class MBBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, surival_pob=0.2, exp_ratio = 6, r = 24):
        super(MBBlock, self).__init__()
        expand_channel = in_channel*exp_ratio
        self.c1 = ConvBlock(in_channel, expand_channel, 1, 1) if exp_ratio > 1 else nn.Identity()
        self.c2 = ConvBlock(expand_channel, expand_channel, kernel_size= kernel_size, stride= stride, groups= expand_channel)
        self.se = SqueezeExcitation(expand_channel, r)
        self.stochastic_depth = StochasticDepth(surival_pob)
        self.c3 = ConvBlock(expand_channel, out_channel, 1, 1, act = False)
        self.add = in_channel == out_channel and stride == 1
    def forward(self, x):
        f = self.c1(x)
        f = self.c2(f)
        f = self.se(f)
        f = self.c3(f)
        if self.add:
          f = self.stochastic_depth(f)
          f = f + x
        return f
# classifier
class Classifier(nn.Module):
    def __init__(self, in_channel, num_class, p):
        super(Classifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channel, num_class, bias = False)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        f = self.pool(x)
        f = self.drop(f)
        f = torch.flatten(f, start_dim= 1)
        f = self.fc(f)
        return f

class EfficientNet(nn.Module):
    def __init__(self, num_class, stochastic_depth_prob = 0.2, version = "b0"):
        super(EfficientNet, self).__init__()
        self.net = nn.ModuleList([])
        self.channels = []
        self.stochastic_depth_prob = stochastic_depth_prob
        stage, phis_value = Config(version)()
        phi, res, p = phis_value
        self._calculate(phi)
        # First stage Conv 3x3
        self.total_stage = self._total_stages(stage)
        self.id_stage = 0
        F, channels, layer, kernel_size, stride, expand = stage[0]
        self._add_layer(3, F, channels, layer, kernel_size, stride)
        #MBConv stage
        for i in range(1, len(stage)-1):
            if i == 1:
                r = 4
            else:
                r=24
            F, channels, layer, kernel_size, stride, expand = stage[i]
            self._add_layer(self.channels[-1], F, channels, layer, kernel_size, stride, expand, r)
        F, channels, layer, kernel_size, stride, expand = stage[-1]
        self._add_layer(self.channels[-1], F, channels, layer, kernel_size, stride)
        # classifier
        self.classifier = Classifier(self.channels[-1], num_class, p)
    def _calculate(self, phi, alpha = 1.2, beta = 1.1):
        self.depth = alpha**phi
        self.width = beta**phi

    def _update_feature(self, channel, layer):
        return int(channel*self.width), int(layer*self.depth)

    def _add_layer(self, in_channels, F, out_channels, layer, kernel_size, stride, *args):
        out_channels, layer = self._update_feature(out_channels, layer)
        if layer == 1:
            if F == MBBlock:
              surival_pob = self.stochastic_depth_prob*float(self.id_stage)/self.total_stage
              self.net.append(F(in_channels, out_channels, kernel_size, stride, surival_pob, *args))
              self.id_stage +=1
            else:
              self.net.append(F(in_channels, out_channels, kernel_size, stride, *args))
        else:
            if F == MBBlock:
              surival_pob = self.stochastic_depth_prob*float(self.id_stage)/self.total_stage
              self.net.append(F(in_channels,out_channels, kernel_size, stride, surival_pob, *args))
              self.id_stage +=1
              for _ in range(layer-1):
                  surival_pob = self.stochastic_depth_prob*float(self.id_stage)/self.total_stage
                  self.net.append(F(out_channels, out_channels, kernel_size, 1, surival_pob, *args))
                  self.id_stage +=1
            else:
              self.net.append(F(in_channels,out_channels, kernel_size, stride, *args))
              for _ in range(layer-1):
                self.net.append(F(out_channels, out_channels, kernel_size, 1, *args))
        self.channels.append(out_channels)
    def _total_stages(self, stage):
      total = 0
      for _, channel, layer, _, _, _ in stage:
        _, layer = self._update_feature(channel, layer)
        total+=layer
      return total
    def forward(self, x):
        for F in self.net:
            x = F(x)
        x = self.classifier(x)
        return x
# config model
class Config:
    def __init__(self, version):
        self.version = version
        self.stage  = [
    # Operator(F), Channels[out_channels], Layer(layer), Kernel(kernel_size), Stride(stride), Expandsion(expand)
            [ConvBlock, 32, 1, 3, 2, 1],
            [MBBlock, 16, 1, 3, 1, 1],
            [MBBlock, 24, 2, 3, 2, 6],
            [MBBlock, 40, 2, 5, 2, 6],
            [MBBlock, 80, 3, 3, 2, 6],
            [MBBlock, 112, 3, 5, 1, 6],
            [MBBlock, 192, 4, 5, 2, 6],
            [MBBlock, 320, 1, 3, 1, 6],
            [ConvBlock, 1280, 1, 1, 1, 0]
        ]
        self.phi_values = {# tuple of: (phi_value, resolution, drop_rate)
            "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
            "b1": (0.5, 240, 0.2),
            "b2": (1, 260, 0.3),
            "b3": (2, 300, 0.3),
            "b4": (3, 380, 0.4),
            "b5": (4, 456, 0.4),
            "b6": (5, 528, 0.5),
            "b7": (6, 600, 0.5),
        }
    def __call__(self):
        phis = self.phi_values[self.version]
        stage = self.stage
        return stage, phis
    # 224, 240, 260, 300, 380, 456, 528, 600
