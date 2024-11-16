""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce
from operator import mul
from einops import rearrange
import cv2 as cv
from scipy.ndimage import convolve


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.bn1 = nn.BatchNorm3d(dim)  # 20240528
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.bn2 = nn.BatchNorm3d(dim)  # 20240528

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(x))  # 20240528
        # return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        # print(self.num_heads)
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = shortcut + self.drop_path(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.bn1(x)  # 20240528
        x = rearrange(x, 'b c d h w -> b d h w c')

        # self.bn2 = nn.BatchNorm3d(x.shape[1])
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            # x = x + self.forward_part2(x)
            x = x + self.forward_part2(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
            x = self.bn2(x)
            x = rearrange(x, 'b c d h w -> b d h w c')

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        # patch merging layer
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        # x=x.permute(0,4,1,2,3)
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class Fuse(nn.Module):
    """ feature fusion. 可以采用 cross attention-based feature fusion   cross attention gate
    Args:
        x1
        x2
    """

    def __init__(self, x1, x2, patch_size=(2, 1, 1), in_channels=None, out_channels=None):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.out_chans = out_channels

        self.bn = nn.BatchNorm3d

        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x1, x2):
        """Forward function."""
        x = self.proj(x1 - x2)
        x = self.relu(x)
        return x


class MultiHeadGate0(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(MultiHeadGate0, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        # 初始化多个门控线性层
        self.gates = nn.ModuleList([nn.Linear(in_dim, int(in_dim / 2)) for _ in range(num_heads)])
        self.final_gate = nn.Linear(int(in_dim * num_heads / 2), int(in_dim / 2))

    def forward(self, x):
        # print(self.in_dim)
        original_shape = x.shape
        batch_size, dim1, dim2, dim3, dim4 = original_shape
        # Reshape x to apply nn.Linear on dim2
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # 将输入特征投影到每个子空间中，并计算每个子空间的门控权重
        gate_outputs = [torch.sigmoid(gate(x)) for gate in self.gates]

        # 将多个子空间的门控权重拼接在一起
        concatenated_gates = torch.cat(gate_outputs, dim=-1)  # size=(B, D, H * num_heads)

        # 最终门控机制，将多个门控头的输出线性变换到与输入维度相同的维度
        output = self.final_gate(concatenated_gates)
        output = output.permute(0, 4, 1, 2, 3)

        return output


class DualSwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        layer_outputs_x = []
        for i, layer in enumerate(self.layers):
            # print(i)
            x = layer(x.contiguous())
            if i != len(self.layers) - 1:
                layer_outputs_x.append(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        return x, layer_outputs_x


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 24 * 8)
        self.fc21 = nn.Linear(24 * 8, latent_dim)  # for mu
        self.fc22 = nn.Linear(24 * 8, latent_dim)  # for logvar

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def tnn(self, xx):
        """
        计算五维张量X的张量核范数
        X的尺度为[batch_size, frames, height, width, channels]
        """
        batch_size, frames, height, width, channels = xx.shape
        tnn_value = []
        for i in range(batch_size):
            x = xx[i]  # [frames, height, width, channels]
            x = rearrange(x, 'c d h w -> d h w c')
            x = rearrange(x, 'd h w c -> (d h w) c')
            tnn = torch.trace(torch.sqrt(x.t().mm(x))) + 1e-8
            tnn_value.append(tnn)
        tnn_value = torch.log(torch.stack(tnn_value) + 1)
        bkg_tnn = torch.sum(tnn_value) / batch_size
        return bkg_tnn

    def forward(self, x):
        x = rearrange(x, 'b c d h w -> b d h w c')
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        loss_vae = self.tnn(mu + logvar)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=4)
        loss_kl = torch.mean(loss_kl)
        z = self.decoder(z)
        z = rearrange(z, 'b d h w c -> b c d h w')
        return z, loss_kl, loss_vae


class SkipConnection(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, scale_factor=1):
        super(SkipConnection, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # 3D point-wise conv
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.up = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_relu(x)
        x = self.up(x)
        return x


class MultiHeadGate(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(MultiHeadGate, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.gates = nn.ModuleList([nn.Linear(in_dim, int(in_dim / 2)) for _ in range(len(num_heads))])
        self.final_gate = nn.Linear(int(in_dim * len(num_heads) / 2), int(in_dim / 2))

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        gate_outputs = [torch.sigmoid(gate(x)) for gate in self.gates]
        concatenated_gates = torch.cat(gate_outputs, dim=-1)  # size=(B, D, H * num_heads)
        output = self.final_gate(concatenated_gates)
        output = output.permute(0, 4, 1, 2, 3)

        return output


def compute_means(padded_img, kernel):
    return convolve(padded_img, kernel, mode='constant', cval=0.0)


class Decoder(nn.Module):
    def __init__(self, embed_dim=96, depths=None, in_channels=None, patch_size=(2, 4, 4), hidden_channels=None,
                 out_channels=None, scale_factor=2, num_heads=[3, 6, 12], device=None):
        super(Decoder, self).__init__()

        self.num_layers = len(depths)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.device = device

        self.DecoderBlock1 = nn.Sequential(
            nn.Conv3d(in_channels=int(2 * embed_dim * 2 ** (self.num_layers - 2)),
                      out_channels=int(embed_dim * 2 ** (self.num_layers - 2)), kernel_size=1),
            nn.ConvTranspose3d(in_channels=int(embed_dim * 2 ** (self.num_layers - 2)),
                               out_channels=int(embed_dim * 2 ** (self.num_layers - 2)), kernel_size=(1, 2, 2),
                               stride=(1, 2, 2)),
            nn.Conv3d(in_channels=int(embed_dim * 2 ** (self.num_layers - 2)),
                      out_channels=int(embed_dim * 2 ** (self.num_layers - 2)), kernel_size=1),
            nn.BatchNorm3d(int(embed_dim * 2 ** (self.num_layers - 2))),
            nn.LeakyReLU(inplace=True)
        )

        self.DecoderBlock2 = nn.Sequential(
            nn.Conv3d(in_channels=int(embed_dim * 2 ** (self.num_layers - 2)),
                      out_channels=int(embed_dim * 2 ** (self.num_layers - 3)), kernel_size=1),
            nn.ConvTranspose3d(in_channels=int(embed_dim * 2 ** (self.num_layers - 3)),
                               out_channels=int(embed_dim * 2 ** (self.num_layers - 3)), kernel_size=(1, 2, 2),
                               stride=(1, 2, 2)),
            nn.Conv3d(in_channels=int(embed_dim * 2 ** (self.num_layers - 3)),
                      out_channels=int(embed_dim * 2 ** (self.num_layers - 3)), kernel_size=1),
            nn.BatchNorm3d(int(embed_dim * 2 ** (self.num_layers - 3))),
            nn.LeakyReLU(inplace=True)
        )

        self.DecoderBlock3 = nn.Sequential(
            nn.Conv3d(in_channels=int(embed_dim * 2 ** (self.num_layers - 3)),
                      out_channels=int(embed_dim * 2 ** (self.num_layers - 4)), kernel_size=1),
            nn.ConvTranspose3d(in_channels=int(embed_dim * 2 ** (self.num_layers - 4)),
                               out_channels=1, kernel_size=(2, 2, 2),
                               stride=(2, 2, 2)),
            nn.Conv3d(in_channels=1,
                      out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

        self.SkipConnection1 = SkipConnection(in_channels=int(embed_dim * 2 ** (self.num_layers - 3)),
                                              out_channels=int(embed_dim * 2 ** (self.num_layers - 3)), scale_factor=1)
        self.SkipConnection2 = SkipConnection(in_channels=int(embed_dim * 2 ** (self.num_layers - 2)),
                                              out_channels=int(embed_dim * 2 ** (self.num_layers - 2)), scale_factor=1)
        self.SkipConnection3 = SkipConnection(in_channels=int(embed_dim * 2 ** (self.num_layers - 1)),
                                              out_channels=int(embed_dim * 2 ** (self.num_layers - 1)), scale_factor=1)

        self.conv1x1 = nn.Conv3d(in_channels=int(embed_dim / 2),
                                 out_channels=1, kernel_size=1)
        self.bn = nn.BatchNorm3d(1)
        self.gate3 = MultiHeadGate(in_dim=int(embed_dim * 2 ** (self.num_layers - 2)), num_heads=num_heads)  # 使用多头门控机制
        self.gate2 = MultiHeadGate(in_dim=int(embed_dim * 2 ** self.num_layers), num_heads=num_heads)  # 使用多头门控机制
        self.gate1 = MultiHeadGate(in_dim=int(embed_dim * 2 ** (self.num_layers - 1)), num_heads=num_heads)  # 使用多头门控机制

    def mean_block(self, img, start_row, start_col):
        return np.mean(img[start_row:start_row + 5, start_col:start_col + 5], axis=(2, 3))

    def calculate_LCM(self, x):
        B, C, D, H, W = x.shape
        x = x.cpu().detach().numpy()
        k = 7
        for m in range(B):
            for n in range(D):
                xtmp = x[m, 0, n, :, :]
                xtmp = 255 * (xtmp - xtmp.min()) / (xtmp.max() - xtmp.min())
                xtmp = cv.GaussianBlur(xtmp, (3, 3), 1)
                padded_img = np.pad(xtmp, ((k, k), (k, k)), mode='constant', constant_values=0)
                kernel_center = np.ones((5, 5)) / 25
                kernel_side = np.ones((5, 5)) / 25
                offsets = [
                    (-6, -6), (-6, -2), (-6, 4),
                    (-2, 4), (4, 4), (4, -2),
                    (4, -6), (-2, -6)
                ]
                mean0 = compute_means(padded_img, kernel_center)[k:-k, k:-k]
                means = np.zeros((H, W, 8))
                for i, (row_offset, col_offset) in enumerate(offsets):
                    means[:, :, i] = compute_means(
                        padded_img[k + row_offset:k + row_offset + H, k + col_offset:k + col_offset + W],
                        kernel_side)
                max_means = np.max(means, axis=2)
                LCM = np.where(mean0 > 1.1 * max_means, (mean0 - max_means) * (mean0 / max_means), 0)
                LCM = (LCM - LCM.min()) / (LCM.max() - LCM.min()+ 0.0001)
                x[m, 0, n, :, :] = LCM

        image = torch.tensor(x)
        return image.to(self.device)

    def forward(self, feature, x):
        x1, x2, x3 = x
        x1 = self.SkipConnection1(x1)
        x2 = self.SkipConnection2(x2)
        x3 = self.SkipConnection3(x3)

        gate_input = torch.cat((feature, x3), dim=1)
        gate_values = self.gate2(gate_input)
        feature = gate_values * feature + (1 - gate_values) * x3
        feature = self.DecoderBlock1(feature)

        gate_input = torch.cat((feature, x2), dim=1)
        gate_values = self.gate1(gate_input)
        feature = gate_values * feature + (1 - gate_values) * x2
        feature = self.DecoderBlock2(feature)

        gate_input = torch.cat((feature, x1), dim=1)
        gate_values = self.gate3(gate_input)
        feature = gate_values * feature + (1 - gate_values) * x1
        x = self.DecoderBlock3(feature)
        #  # true during training
        # lcm = self.calculate_LCM(x)
        # x = x / (lcm + 0.0001)
        # x = (x - x.min()) / (x.max() - x.min())
        return x


class SwinTransformerDecoder(nn.Module):
    def __init__(self,
                 pretrained=None, pretrained2d=None, patch_size1=(1, 2, 2), patch_size2=(2, 4, 4),
                 patch_size3=(4, 8, 8), in_chans=1, embed_dim=96,
                 depths1=[2, 2], num_heads1=[3, 6], depths2=[2, 2, 2], num_heads2=[3, 6, 12],
                 depths3=[2, 2, 2, 2], num_heads3=[3, 6, 12, 24],
                 window_size1=(1, 3, 3), window_size2=(2, 7, 7), window_size3=(4, 15, 15),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=False,
                 frozen_stages=-1, use_checkpoint=False, hidden_channels=None, out_channels=1,
                 scale_factor=2, x1=None, x2=None,
                 gate_head=4, device=None):
        super(SwinTransformerDecoder, self).__init__()
        self.x1 = x1
        self.x2 = x2
        self.gate_head = gate_head
        self.in_channels2 = int(embed_dim * 2 ** (len(depths2) - 1))
        self.depths = depths2
        self.DualSwinTransformer3D1 = DualSwinTransformer3D(pretrained, pretrained2d,
                                                            patch_size1, in_chans, embed_dim, depths1, num_heads1,
                                                            window_size1, mlp_ratio,
                                                            qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                                                            drop_path_rate, norm_layer, patch_norm,
                                                            frozen_stages, use_checkpoint)
        self.DualSwinTransformer3D2 = DualSwinTransformer3D(pretrained, pretrained2d,
                                                            patch_size2, in_chans, embed_dim, depths2, num_heads2,
                                                            window_size2, mlp_ratio,
                                                            qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                                                            drop_path_rate, norm_layer, patch_norm,
                                                            frozen_stages, use_checkpoint)
        self.DualSwinTransformer3D3 = DualSwinTransformer3D(pretrained, pretrained2d,
                                                            patch_size3, in_chans, embed_dim, depths3, num_heads3,
                                                            window_size3, mlp_ratio,
                                                            qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                                                            drop_path_rate, norm_layer, patch_norm,
                                                            frozen_stages, use_checkpoint)

        self.Decoder = Decoder(embed_dim, depths2, self.in_channels2, patch_size2, hidden_channels, out_channels,
                               scale_factor, num_heads2, device)
        self.vae = VAE(input_dim=self.in_channels2, latent_dim=48, output_dim=self.in_channels2)

        self.down1 = nn.Conv3d(in_channels=embed_dim, out_channels=self.in_channels2, kernel_size=(1, 3, 3),
                               stride=(1, 2, 2), padding=(0, 1, 1))

        self.con1x1_feature1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels2, out_channels=self.in_channels2, kernel_size=(1, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(self.in_channels2),
            nn.LeakyReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.ConvTranspose3d(in_channels=embed_dim * 4, out_channels=embed_dim * 2,
                                                    kernel_size=(1, 2, 2),
                                                    stride=(1, 2, 2)),
                                 nn.Conv3d(in_channels=embed_dim * 2, out_channels=embed_dim * 2, kernel_size=(1, 3, 3),
                                           stride=(1, 1, 1), padding=(0, 1, 1)),
                                 nn.BatchNorm3d(embed_dim * 2))

        self.con1x1_feature3 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels2, out_channels=self.in_channels2, kernel_size=(1, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(self.in_channels2),
            nn.LeakyReLU(inplace=True))

        self.up21 = nn.Sequential(nn.ConvTranspose3d(in_channels=self.in_channels2,
                                                     out_channels=int(self.in_channels2 / 2),
                                                     kernel_size=(1, 2, 2),
                                                     stride=(1, 2, 2)),
                                  nn.Conv3d(in_channels=int(self.in_channels2 / 2), out_channels=int(self.in_channels2 / 2),
                                            kernel_size=(1, 3, 3),
                                            stride=(1, 1, 1), padding=(0, 1, 1)),
                                  nn.BatchNorm3d(int(self.in_channels2 / 2)))

        self.con1x1_feature21 = nn.Sequential(
            nn.Conv3d(in_channels=int(self.in_channels2 / 2), out_channels=int(self.in_channels2 / 2),
                      kernel_size=(1, 3, 3),stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(int(self.in_channels2 / 2)),
            nn.LeakyReLU(inplace=True))

        self.up31 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.in_channels2, out_channels=int(self.in_channels2 / 4),
                               kernel_size=(1, 4, 4),
                               stride=(1, 4, 4)),
            nn.Conv3d(in_channels=int(self.in_channels2 / 4), out_channels=int(self.in_channels2 / 4),
                      kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(int(self.in_channels2 / 4)),
            nn.LeakyReLU(inplace=True))

        self.con1x1_feature31 = nn.Sequential(
            nn.Conv3d(in_channels=int(self.in_channels2 / 4), out_channels=int(self.in_channels2 / 4),
                      kernel_size=(1, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(int(self.in_channels2 / 4)),
            nn.LeakyReLU(inplace=True))

        self.up32 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=embed_dim * 4, out_channels=self.in_channels2, kernel_size=(1, 2, 2),
                               stride=(1, 2, 2)),
            nn.Conv3d(in_channels=self.in_channels2, out_channels=self.in_channels2,
                      kernel_size=(1, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(self.in_channels2),
            nn.LeakyReLU(inplace=True))

        self.con1x1_feature32 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels2, out_channels=self.in_channels2,
                      kernel_size=(1, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(self.in_channels2),
            nn.LeakyReLU(inplace=True))

        self.down_sample_plan3 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=48,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=48,
                out_channels=96,
                kernel_size=(3, 3, 3),
                stride=(2, 4, 4),
                padding=(1, 1, 1)),
            nn.BatchNorm3d((embed_dim * 2)),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):

        feature1, layer_outputs_x1 = self.DualSwinTransformer3D1(x)

        feature2, layer_outputs_x2 = self.DualSwinTransformer3D2(x)

        feature3, layer_outputs_x3 = self.DualSwinTransformer3D3(x)

        local_feature = self.down_sample_plan3(x)

        feature1 = self.down1(feature1)
        feature1 = self.con1x1_feature1(feature1)

        feature3 = self.up3(feature3)
        feature3 = self.con1x1_feature3(feature3)

        layer_outputs_x2[0] = self.up21(layer_outputs_x2[0])
        layer_outputs_x2[0] = self.con1x1_feature21(layer_outputs_x2[0])

        layer_outputs_x3[0] = self.up31(layer_outputs_x3[0])
        layer_outputs_x3[0] = self.con1x1_feature31(layer_outputs_x3[0])

        layer_outputs_x3[1] = self.up32(layer_outputs_x3[1])
        layer_outputs_x3[1] = self.con1x1_feature32(layer_outputs_x3[1])

        combined_outputs = [layer_outputs_x3[0], layer_outputs_x2[0], layer_outputs_x3[1]]

        feature, loss_kl, loss_vae = self.vae(feature1 + feature2 + feature3 + local_feature)
        pred = self.Decoder(feature, combined_outputs)

        return x, pred, loss_kl, loss_vae


def mySwinTransformerDecoder(*args, **kwargs):
    model = SwinTransformerDecoder(*args, **kwargs)
    return model


if __name__ == "__main__":
    instant_model = SwinTransformerDecoder()
    print(instant_model)
