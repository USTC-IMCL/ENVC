import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c, ks=3):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(c, c, ks, 1, ks//2),
            nn.ReLU(),
            nn.Conv2d(c, c, ks, 1, ks//2)
        )

    def forward(self, input):
        return input + self.res_block(input)


class ResBlocks(nn.Module):
    def __init__(self, c, ks=3):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResBlock(c, ks),
            ResBlock(c, ks),
            ResBlock(c, ks)
        )

    def forward(self, input):
        return input + self.res_blocks(input)


class TransformEncoder(nn.Module):
    def __init__(self, C, N, M, k):
        super().__init__()
        self.m = nn.ModuleDict()
        self.m.down1 = nn.Conv2d(C, N, 4, 2, 1)
        self.m.rb1 = ResBlocks(N, ks=k)
        self.m.down2 = nn.Conv2d(N, N, 4, 2, 1)
        self.m.rb2 = ResBlocks(N, ks=k)
        self.m.down3 = nn.Conv2d(N, N, 4, 2, 1)
        self.m.rb3 = ResBlocks(N, ks=k)
        self.m.down4 = nn.Conv2d(N, M, 4, 2, 1)

    def forward(self, x):
        x = self.m.down1(x)
        x = self.m.rb1(x)
        x = self.m.down2(x)
        x = self.m.rb2(x)
        x = self.m.down3(x)
        x = self.m.rb3(x)
        x = self.m.down4(x)
        return x


class TransformDecoder(nn.Module):
    def __init__(self, C, N, M, k):
        super().__init__()
        self.m = nn.ModuleDict()
        self.m.up4 = nn.ConvTranspose2d(M, N, 4, 2, 1)
        self.m.rb3 = ResBlocks(N, ks=k)
        self.m.up3 = nn.ConvTranspose2d(N, N, 4, 2, 1)
        self.m.rb2 = ResBlocks(N, ks=k)
        self.m.up2 = nn.ConvTranspose2d(N, N, 4, 2, 1)
        self.m.rb1 = ResBlocks(N, ks=k)
        self.m.up1 = nn.ConvTranspose2d(N, C, 4, 2, 1)

    def forward(self, x):
        x = self.m.up4(x)
        x = self.m.rb3(x)
        x = self.m.up3(x)
        x = self.m.rb2(x)
        x = self.m.up2(x)
        x = self.m.rb1(x)
        x = self.m.up1(x)
        return x


class PreTransformEncoder(nn.Module):
    def __init__(self, C, N, k, lmin=0, lmax=1):
        super().__init__()
        self.m = nn.ModuleDict()
        for i in range(lmin, lmax):
            c = C if i + 1 == 1 else N
            self.m['down' + str(i + 1)] = nn.Conv2d(c, N, 4, 2, 1)
            self.m['rb' + str(i + 1)] = ResBlocks(N, ks=k)

    def forward(self, x):
        for m in self.m.values():
            x = m(x)
        return x


class PreTransformDecoder(nn.Module):
    def __init__(self, C, N, k, lmin=0, lmax=1):
        super().__init__()
        self.m = nn.ModuleDict()
        for i in range(lmin, lmax)[::-1]:
            c = C if i + 1 == 1 else N
            self.m['rb' + str(i + 1)] = ResBlocks(N, ks=k)
            self.m['up' + str(i + 1)] = nn.ConvTranspose2d(N, c, 4, 2, 1)

    def forward(self, x):
        for m in self.m.values():
            x = m(x)
        return x


class FeatureTransformEncoder(nn.Module):
    def __init__(self, N, M, k, lmin=1, lmax=4):
        super().__init__()
        self.m = nn.ModuleDict()
        for i in range(lmin, lmax - 1):
            self.m['down' + str(i + 1)] = nn.Conv2d(N, N, 4, 2, 1)
            self.m['rb' + str(i + 1)] = ResBlocks(N, ks=k)
        self.m.down4 = nn.Conv2d(N, M, 4, 2, 1)

    def forward(self, x):
        for m in self.m.values():
            x = m(x)
        return x


class FeatureTransformDecoder(nn.Module):
    def __init__(self, N, M, k, lmin=1, lmax=4):
        super().__init__()
        self.m = nn.ModuleDict()
        self.m.up4 = nn.ConvTranspose2d(M, N, 4, 2, 1)
        for i in range(lmin, lmax - 1)[::-1]:
            self.m['rb' + str(i + 1)] = ResBlocks(N, ks=k)
            self.m['up' + str(i + 1)] = nn.ConvTranspose2d(N, N, 4, 2, 1)

    def forward(self, x):
        x_list = []
        for m in self.m.values():
            x = m(x)
            if isinstance(m, nn.ConvTranspose2d):
                x_list.append(x)
        return x


class HierTransformDecoder(nn.Module):
    def __init__(self, N, M, k, lmin=1, lmax=4):
        super().__init__()
        self.m = nn.ModuleDict()
        self.m.up4 = nn.ConvTranspose2d(M, N, 4, 2, 1)
        for i in range(lmin, lmax - 1)[::-1]:
            self.m['rb' + str(i + 1)] = ResBlocks(N, ks=k)
            self.m['up' + str(i + 1)] = nn.ConvTranspose2d(N, N, 4, 2, 1)

    def forward(self, x):
        x_list = []
        for m in self.m.values():
            x = m(x)
            if isinstance(m, nn.ConvTranspose2d):
                x_list.append(x)
        return x, x_list


class HierarchicalDownsampling(nn.Module):
    def __init__(self, c, k, num_layer):
        super().__init__()
        self.m = nn.ModuleDict()
        for i in range(1, num_layer):
            self.m['down' + str(i + 1)] = nn.Conv2d(c, c, 4, 2, 1)
            self.m['rb' + str(i + 1)] = ResBlocks(c, ks=k)

    def forward(self, x):
        x_list = []
        x_list.append(x)
        for m in self.m.values():
            x = m(x)
            if isinstance(m, ResBlocks):
                x_list.append(x)
        return x_list


class HyperEncoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.m = nn.ModuleDict()
        self.m.lrelu = nn.LeakyReLU()
        self.m.conv1 = nn.Conv2d(M, N, 3, 1, 1)
        self.m.down1 = nn.Conv2d(N, N, 4, 2, 1)
        self.m.down2 = nn.Conv2d(N, N, 4, 2, 1)

    def forward(self, x):
        x = self.m.conv1(x)
        x = self.m.lrelu(x)
        x = self.m.down1(x)
        x = self.m.lrelu(x)
        x = self.m.down2(x)
        return x


class HyperDecoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.m = nn.ModuleDict()
        self.m.lrelu = nn.LeakyReLU()
        self.m.up2 = nn.ConvTranspose2d(N, N, 4, 2, 1)
        self.m.up1 = nn.ConvTranspose2d(N, M, 4, 2, 1)
        self.m.conv1 = nn.ConvTranspose2d(M, 2*M, 3, 1, 1)

    def forward(self, x):
        x = self.m.up2(x)
        x = self.m.lrelu(x)
        x = self.m.up1(x)
        x = self.m.lrelu(x)
        x = self.m.conv1(x)
        return x


class ParameterModelGMM(nn.Module):
    def __init__(self, M, ctx_depth):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ctx_depth, 4*M, 1, 1, 0), nn.LeakyReLU(),
            ResBlock(4*M, ks=1), nn.LeakyReLU(),
            nn.Conv2d(4*M, 9*M, 1, 1, 0))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, ctx):
        para = self.conv(ctx)
        para = para.chunk(9, dim=1)
        weight = para[0:3]
        weight = self.softmax(torch.stack(weight, 0))
        weight = [weight[i] for i in range(3)]
        return weight, para[3:6], para[6:9]


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2:] = 0
        self.mask[:, :, kH // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)