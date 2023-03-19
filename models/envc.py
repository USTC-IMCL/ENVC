import abc
import math
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import ResBlock, ResBlocks, TransformEncoder, TransformDecoder, \
    HyperEncoder, HyperDecoder, ParameterModelGMM, MaskedConv2d, \
    PreTransformEncoder, HierarchicalDownsampling, FeatureTransformEncoder, \
    FeatureTransformDecoder, PreTransformDecoder
from .layers_entropy import GaussianEntropy, FactorizedEntropy
from .layers_warp import MultiScaleWarpLayer

__all__ = [
    "ENVCwAR"
]
Tensor = torch.Tensor


class IntraInterCodingModelBase(nn.Module, metaclass=abc.ABCMeta):
    frame_type_table = ["I", "P"]

    def __init__(self):
        super().__init__()
        self.register_buffer("compression", torch.zeros(1, dtype=torch.bool))

    def forward_one_frame(self, x: Tensor, ref_buffer: Dict, frame_type: str,
                          **kwargs):
        assert frame_type in self.frame_type_table
        if frame_type == "I":
            return self.forward_intra(x, **kwargs)
        else:
            return self.forward_inter(x, ref_buffer, **kwargs)

    @abc.abstractmethod
    def forward_intra(self, x: Tensor, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward_inter(self, x: Tensor, ref_buffer: Dict, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_ready_for_compression(self):
        """Get everything ready before separating the encoder and decoder.
        """
        self.compression.fill_(True)

    @abc.abstractmethod
    def compress(self, x: Tensor, ref_buffer: Dict, **kwargs) -> List[str]:
        """Compress `x` into byte strings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decompress(self, strings: List[str], ref_buffer: Dict, **kwargs) \
            -> Tensor:
        """Decompress `x` given byte strings
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def downscale_factor(self):
        """ Maximum down-scale factor
        """
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    def pre_padding(self, x):
        hgt, wdt = x.shape[2:4]
        factor = self.downscale_factor
        dh = factor * math.ceil(hgt / factor) - hgt
        dw = factor * math.ceil(wdt / factor) - wdt
        x = F.pad(x, (dw // 2, dw // 2 + dw % 2, dh // 2, dh // 2 + dh % 2))
        return x

    def post_cropping(self, x, wdt, hgt):
        factor = self.downscale_factor
        dh = factor * math.ceil(hgt / factor) - hgt
        dw = factor * math.ceil(wdt / factor) - wdt
        return x[..., dh // 2: -(dh // 2 + dh % 2) or None, \
               dw // 2: -(dw // 2 + dw % 2) or None]

    @staticmethod
    def es_bits(prob):
        return -torch.sum(torch.log2(prob))


class ENVCwAR(IntraInterCodingModelBase):
    def __init__(self, Ni=128, Mi=160, N=128, M=160, num_head=4, num_point=4):
        super().__init__()
        k = 3  # the kernel size of resblock
        L = 1  # the resolution level of featrue space, 2^L downsampling ratio

        self.GM = GaussianEntropy()

        # Intra compression model
        self.I = nn.ModuleDict()
        self.I.main_enc = TransformEncoder(3, Ni, Mi, k)
        self.I.main_dec = TransformDecoder(3, Ni, Mi, k)
        self.I.hyper_enc = HyperEncoder(Ni, Mi)
        self.I.hyper_dec = HyperDecoder(Ni, Mi)
        self.I.py_PM = ParameterModelGMM(Mi, ctx_depth=4 * M)
        self.I.pz_FM = FactorizedEntropy(Ni)
        self.I.iy_AR = MaskedConv2d(Mi, Mi * 2, 5, 1, 2)

        # Inter compression model
        self.P = nn.ModuleDict()
        self.P.motion_enc_pre = PreTransformEncoder(3, N, k, lmin=0, lmax=L)
        self.P.hier_down = HierarchicalDownsampling(N, k, 4 - L)
        self.P.concat_conv = nn.Sequential(
            nn.Conv2d(2 * N, N, 3, 1, 1),
            ResBlock(N, ks=k)
        )
        self.P.motion_enc_ft = FeatureTransformEncoder(N, M // 2, k, lmin=L,
                                                       lmax=4)
        self.P.motion_dec_ft = FeatureTransformDecoder(N // 2, M // 2, k,
                                                       lmin=L, lmax=4)
        self.P.process_net = nn.Sequential(
            nn.Conv2d(N, N, 3, 1, 1),
            ResBlocks(N, ks=k)
        )

        self.P.my_PM = ParameterModelGMM(M // 2, ctx_depth=M)
        self.P.my_AR = MaskedConv2d(M // 2, M, 5, 1, 2)

        self.P.res_enc_pre = ResBlocks(N, k)
        self.P.res_dec_pre = PreTransformDecoder(3, N, k)
        self.P.res_enc_ft = FeatureTransformEncoder(N, M, k)
        self.P.res_dec_ft = FeatureTransformDecoder(N, M, k)

        self.P.res_hyper_enc = HyperEncoder(N, M)
        self.P.res_hyper_dec = HyperDecoder(N, M)
        self.P.ry_PM = ParameterModelGMM(M, ctx_depth=4 * M)
        self.P.rz_FM = FactorizedEntropy(N)
        self.P.ry_AR = MaskedConv2d(M, M * 2, 5, 1, 2)

        self.P.warp = MultiScaleWarpLayer(
            d_model=N,
            d_flow=N // 2,
            d_ffn=N * 2,
            dropout=0.1,
            n_levels=4 - L,
            n_heads=num_head,
            n_points=num_point
        )

    def forward_intra(self, x, ref_buffer=None, **kwargs):
        y = self.I.main_enc(x)
        z = self.I.hyper_enc(y)
        y_hat = y.round()
        z_hat = z.round()

        y_hyper = self.I.hyper_dec(z_hat)
        y_ar = self.I.iy_AR(y_hat)
        y_ctx = torch.cat([y_ar, y_hyper], dim=1)
        y_weight, y_mean, y_scale = self.I.py_PM(y_ctx)
        x_hat = self.I.main_dec(y_hat)

        y_bits = self.es_bits(self.GMM(y_hat, y_weight, y_mean, y_scale))
        z_bits = self.es_bits(self.I.pz_FM(z_hat))

        # update reference buffer
        ref_buffer = {"x_hat": x_hat}
        return x_hat, [y_bits, z_bits], ref_buffer

    def forward_inter(self, x, ref_buffer, **kwargs):
        x_ref = ref_buffer["x_hat"]
        # from pixel space to feature space
        fx_cur = self.P.motion_enc_pre(x)
        fx_ref = self.P.motion_enc_pre(x_ref)

        # motion encoding
        fx_cat = torch.cat([fx_ref, fx_cur], dim=1)
        my = self.P.motion_enc_ft(self.P.concat_conv(fx_cat))
        my = torch.clamp(my, min=-20.0, max=20.0)
        my_hat = my.round()

        # reference feature pyramid
        fx_refs = self.P.hier_down(fx_ref)

        # motion decoding
        flow = self.P.motion_dec_ft(my_hat)
        my_ar = self.P.my_AR(my_hat)
        my_weight, my_mean, my_scale = self.P.my_PM(my_ar)

        # estimate motion bits
        my_bits = self.es_bits(self.GMM(my_hat, my_weight, my_mean, my_scale))
        mz_bits = self.es_bits(torch.ones(1).to(my_hat))

        # cross-scale prediction
        fx_bar = self.cross_scale_prediction(fx_refs, flow)
        fx_bar_compensate = self.P.process_net(fx_bar)

        # feature residual encoding
        fx = self.P.res_enc_pre(fx_cur)
        fx_bar = self.P.res_enc_pre(fx_bar_compensate)
        ft_res = fx - fx_bar
        ry = self.P.res_enc_ft(ft_res)
        rz = self.P.res_hyper_enc(ry)
        ry_hat = ry.round()
        rz_hat = rz.round()

        # feature residual decoding
        ft_res_hat = self.P.res_dec_ft(ry_hat)
        fx_hat = ft_res_hat + fx_bar
        x_hat = self.P.res_dec_pre(fx_hat)
        ry_hyper = self.P.res_hyper_dec(rz_hat)
        ry_ar = self.P.ry_AR(ry_hat)
        ry_weight, ry_mean, ry_scale = self.P.ry_PM(
            torch.cat([ry_ar, ry_hyper], dim=1))

        # estimate residual bits
        ry_bits = self.es_bits(self.GMM(ry_hat, ry_weight, ry_mean, ry_scale))
        rz_bits = self.es_bits(self.P.rz_FM(rz_hat))

        # update reference buffer
        ref_buffer = {"x_hat": x_hat}
        return x_hat, [my_bits, mz_bits, ry_bits, rz_bits], ref_buffer

    def cross_scale_prediction(self, fx_refs, flow):
        src_flatten = []
        spatial_shapes = []
        for lvl, (src,) in enumerate(zip(fx_refs)):
            b, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # B, HW, C
            src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, 1)
        flow_flatten = flow.flatten(2).transpose(1, 2)
        # [[64, 64], [32, 32], [16,16]]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long)
        spatial_shapes = spatial_shapes.to(src_flatten.device)

        # level_start_index = [0, 4096, 5120]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))

        # warping
        out = self.P.warp(src_flatten, flow_flatten, spatial_shapes,
                          level_start_index)
        b, c, h, w = fx_refs[0].shape
        fx_out = out.transpose(1, 2).view(b, c, h, w)
        return fx_out

    def GMM(self, x, weight, mean, scale):
        assert len(weight) == len(mean) == len(scale)
        likelihood = [weight[i] * self.GM(x, mean[i], scale[i]) for i in
                      range(len(weight))]
        return sum(likelihood)

    def initialize_flow(self, feature):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = feature.shape
        coords = self.coords_grid(N, H, W).to(feature.device)
        return coords

    @staticmethod
    def coords_grid(batch, ht, wd):
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    @staticmethod
    def build_frame_type_list(n_frames, intra_period):
        intra_indexes = np.arange(0, n_frames, intra_period)
        frame_type_list = []
        for idx in range(n_frames):
            if idx in intra_indexes:
                frame_type_list.append("I")
            else:
                frame_type_list.append("P")
        return frame_type_list

    @property
    def downscale_factor(self):
        """ Maximum down-scale factor
        """
        return 2 ** (4 + 2)

    def get_ready_for_compression(self):
        pass

    def compress(self):
        pass

    def decompress(self):
        pass
