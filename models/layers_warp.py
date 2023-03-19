import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.msda.functions import MSDeformAttnFunction
from third_party.msda.modules import MSDeformAttn


class MSDeformWarp(MSDeformAttn):
    def __init__(self, d_model=256, d_flow=128, n_levels=4, n_heads=8,
                 n_points=4):
        super().__init__(d_model=d_model, n_levels=n_levels, n_heads=n_heads,
                         n_points=n_points)
        self.sampling_offsets = nn.Linear(
            d_flow, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(
            d_flow, n_heads * n_levels * n_points)
        self._reset_parameters()

    def forward(self, flow, ref_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        nh = self.n_heads
        dm = self.d_model
        nl = self.n_levels
        np = self.n_points

        N, Len_q, _ = flow.shape
        N, LeNin, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == LeNin

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, LeNin, nh, dm // nh)
        sampling_offsets = self.sampling_offsets(flow).view(N, Len_q, nh, nl, np, 2)
        attention_weights = self.attention_weights(flow).view(N, Len_q, nh, nl * np)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, nh, nl, np)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if ref_points.shape[-1] == 2:
            offset_norm = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                dim=-1
            )
            sampling_locations = ref_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_norm[None, None, None, :, None, :]
        elif ref_points.shape[-1] == 4:
            sampling_locations = ref_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / np * ref_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of ref_points must be 2 or 4')
        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step
        )
        output = self.output_proj(output)
        return output


class MultiScaleWarpLayer(nn.Module):
    def __init__(self, d_model=256, d_flow=128, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.cross_attn = MSDeformWarp(
            d_model, d_flow, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)

        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_ref_points(spatial_shapes, valid_ratios, device):
        hgt, wdt = spatial_shapes[0]
        dtype = torch.float32
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, hgt - 0.5, hgt, dtype=dtype, device=device),
            torch.linspace(0.5, wdt - 0.5, wdt, dtype=dtype, device=device),
            indexing="ij"
        )
        ref_y = ref_y.reshape(-1)[None] / hgt
        ref_x = ref_x.reshape(-1)[None] / wdt
        ref = torch.stack((ref_x, ref_y), -1)
        ref_points = ref[:, :, None] * valid_ratios[:, None]
        return ref_points

    def forward_ffn(self, src):
        out = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return out

    def forward(self, src, flow, spatial_shapes, level_start_index):
        # src.shape = [B, 64*64+32*32+16*16, C]
        # pos.shape = [B, 64*64+32*32+16*16, C]
        # ref_points.shape = [B, 64*64+32*32+16*16, 4-L, 2]
        # spatial_shapes = [[64, 64], [32, 32], [16,16]]
        # level_start_index = [0, 4096, 5120]

        valid_ratios = torch.ones([1, 3, 2], device=src.device)
        ref_points = self.get_ref_points(
            spatial_shapes, valid_ratios, device=src.device)
        src2 = self.cross_attn(
            flow, ref_points, src, spatial_shapes, level_start_index)
        out = self.dropout1(src2)

        out = out + self.forward_ffn(self.norm1(out))
        return out
