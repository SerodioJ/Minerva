# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the THIRD_PARTY_LICENSES/Apache-2.0.txt file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/block.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/drop_path.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/layer_scale.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/mlp.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/patch_embed.py
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py

import os
from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network layer.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    hidden_features : int, optional
        Hidden layer feature dimension.
    out_features : int, optional
        Output feature dimension.
    act_layer : callable, optional
        Activation function.
    drop : float, default 0.0
        Dropout probability.
    bias : bool, default True
        Whether linear projections include bias.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SwiGLU FFN."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# Native PyTorch fallback for SwiGLU
SwiGLU = SwiGLUFFN
# vit = vit_base(patch_size=14, img_size=518, block_chunks=0, layerscale_init=1.0)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    """
    Apply a function recursively to named submodules.

    Parameters
    ----------
    fn : callable
        Function to execute on each submodule.
    module : nn.Module
        Root module.
    name : str, default ""
        Prefix name for submodules.
    depth_first : bool, default True
        Whether to recurse depth-first.
    include_root : bool, default False
        Whether to execute `fn` on the root module itself.

    Returns
    -------
    nn.Module
        Modified root module.
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """
    Initialize ViT layer weights following timm convention for reproducibility.

    Parameters
    ----------
    module : nn.Module
        PyTorch module whose weights are to be initialized.
    name : str, default ""
        Name of the module within the hierarchy.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # This is close to timm's default
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="linear",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, LayerScale):
        nn.init.constant_(module.gamma, module.init_values)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    """Construct a DINOv2 ViT-Small model (embed_dim=384, depth=12, num_heads=6)."""
    depth = kwargs.pop("depth", 12)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        ffn_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    """Construct a DINOv2 ViT-Base model (embed_dim=768, depth=12, num_heads=12)."""
    depth = kwargs.pop("depth", 12)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        ffn_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    """Construct a DINOv2 ViT-Large model (embed_dim=1024, depth=24, num_heads=16)."""
    depth = kwargs.pop("depth", 24)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        ffn_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """Construct a DINOv2 ViT-Giant model (embed_dim=1536, depth=40, num_heads=24)."""
    depth = kwargs.pop("depth", 40)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        ffn_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


# layers.attention
class Attention(nn.Module):
    """
    Standard Multi-Head Self-Attention module.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    num_heads : int, default 8
        Number of attention heads.
    qkv_bias : bool, default False
        Whether query/key/value projections include bias.
    proj_bias : bool, default True
        Whether output linear projection includes bias.
    attn_drop : float, default 0.0
        Dropout rate on attention probabilities.
    proj_drop : float, default 0.0
        Dropout rate on output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """Compute multi-head self-attention."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """
    Memory-efficient multi-head attention using PyTorch scaled dot product attention.
    """

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        """Compute memory-efficient scaled dot product attention."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # PyTorch Native Scaled Dot Product Attention (SDPA)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# layers.mlp
class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block with activation and dropout.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    hidden_features : int, optional
        Hidden layer dimension.
    out_features : int, optional
        Output feature dimension.
    act_layer : callable, default nn.GELU
        Activation layer constructor.
    drop : float, default 0.0
        Dropout probability.
    bias : bool, default True
        Whether linear layers include bias.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP block."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# layers.block
class _Block(nn.Module):
    """
    Transformer block containing self-attention and feed-forward network with LayerScale and stochastic depth.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, default 4.0
        Ratio of MLP hidden dimension to embedding dimension.
    qkv_bias : bool, default False
        Whether attention QKV projections include bias.
    proj_bias : bool, default True
        Whether attention output projection includes bias.
    ffn_bias : bool, default True
        Whether FFN linear layers include bias.
    drop : float, default 0.0
        Dropout rate.
    attn_drop : float, default 0.0
        Attention dropout rate.
    init_values : float, optional
        Initial value for LayerScale gamma parameter.
    drop_path : float, default 0.0
        Stochastic depth drop path rate.
    act_layer : callable, default nn.GELU
        Activation function.
    norm_layer : callable, default nn.LayerNorm
        Normalization layer.
    attn_class : callable, default Attention
        Attention class constructor.
    ffn_layer : callable, default Mlp
        FFN class constructor.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer block."""

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    """
    Apply stochastic depth by dropping residual paths for a random subset of samples in the batch.

    Parameters
    ----------
    x : torch.Tensor
        Input feature tensor.
    residual_func : callable
        Function computing the residual branch.
    sample_drop_ratio : float, default 0.0
        Fraction of batch samples to drop.

    Returns
    -------
    torch.Tensor
        Output tensor with residual added to surviving samples.
    """
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Block(_Block):
    """
    Transformer Block wrapper supporting either a single Tensor or a list of Tensors.
    """

    def forward(
        self, x_or_x_list: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass supporting single Tensor or list of crop Tensors."""
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            out = []
            for x in x_or_x_list:
                out.append(super(Block, self).forward(x))
            return out
        else:
            raise AssertionError


# layers.drop path


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Apply stochastic depth (drop path) per sample.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    drop_prob : float, default 0.0
        Probability of dropping paths.
    training : bool, default False
        Whether currently in training mode.

    Returns
    -------
    torch.Tensor
        Tensor after stochastic depth.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Parameters
    ----------
    drop_prob : float, optional
        Drop path probability.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass applying drop_path."""
        return drop_path(x, self.drop_prob, self.training)


# layers.layer_scale
class LayerScale(nn.Module):
    """
    LayerScale module for scaling residual outputs by learnable diagonal parameters.

    Parameters
    ----------
    dim : int
        Channel dimension.
    init_values : float or torch.Tensor, default 1e-5
        Initial scaling value.
    inplace : bool, default False
        Whether to perform multiplication in-place.
    """

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.init_values = init_values

    def forward(self, x: Tensor) -> Tensor:
        """Scale input tensor by gamma."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# layers.patch_embed
def make_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert an int or tuple to a 2-element tuple."""
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Parameters
    ----------
    img_size : int or tuple of int, default 224
        Input image resolution.
    patch_size : int or tuple of int, default 16
        Patch resolution.
    in_chans : int, default 3
        Number of input channels.
    embed_dim : int, default 768
        Embedding projection dimension.
    norm_layer : callable, optional
        Normalization layer.
    flatten_embedding : bool, default True
        Whether to flatten spatial patch grid to a token sequence.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Convert input image tensor to patch embeddings."""
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        """Estimate FLOP count for patch projection."""
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


# layers.swiglu_ffn
class SwiGLUFFNFused(SwiGLU):
    """
    Fused SwiGLU FFN with 2/3 hidden dimension ratio rounded to multiples of 8.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    hidden_features : int, optional
        Hidden layer dimension.
    out_features : int, optional
        Output feature dimension.
    act_layer : callable, optional
        Activation layer.
    drop : float, default 0.0
        Dropout rate.
    bias : bool, default True
        Whether linear projections include bias.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class DinoVisionTransformer(nn.Module):
    """
    DINOv2 Vision Transformer (ViT) architecture with register tokens and stochastic depth.

    Parameters
    ----------
    img_size : int or tuple of int, default 224
        Input image resolution.
    patch_size : int or tuple of int, default 16
        Patch resolution.
    in_chans : int, default 3
        Number of input image channels.
    embed_dim : int, default 768
        Embedding feature dimension.
    depth : int, default 12
        Number of transformer blocks.
    num_heads : int, default 12
        Number of attention heads.
    ffn_ratio : float, default 4.0
        Expansion ratio for FFN hidden dimension.
    qkv_bias : bool, default True
        Whether QKV projections include bias.
    ffn_bias : bool, default True
        Whether FFN projections include bias.
    proj_bias : bool, default True
        Whether attention projection includes bias.
    drop_path_rate : float, default 0.0
        Stochastic depth drop path rate.
    drop_path_uniform : bool, default False
        Whether to apply uniform drop path rate across all blocks.
    layerscale_init : float, optional
        LayerScale initial diagonal value.
    embed_layer : callable, default PatchEmbed
        Patch embedding constructor.
    act_layer : callable, default nn.GELU
        Activation constructor.
    block_fn : callable, default Block
        Transformer block constructor.
    ffn_layer : {"mlp", "swiglu", "swiglufused", "identity"}, default "mlp"
        Feed-forward network type.
    block_chunks : int, default 1
        Number of chunks to split block sequence for FSDP wrapping.
    num_register_tokens : int, default 0
        Number of additional register tokens.
    interpolate_antialias : bool, default False
        Whether to use antialiasing when interpolating positional encodings.
    interpolate_offset : float, default 0.1
        Offset for coordinate grid interpolation.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        layerscale_init=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            print("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            print("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            print("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                attn_class=MemEffAttention,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=layerscale_init,
            )
            for i in range(depth)
        ]
        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        """Initialize positional embeddings, CLS token, register tokens, and submodules."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        """
        Bicubic interpolation of learned positional embeddings to match input spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input token tensor of shape `(B, N, D)`.
        w : int
            Image pixel width.
        h : int
            Image pixel height.

        Returns
        -------
        torch.Tensor
            Interpolated positional encoding tensor.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Optional[Tensor] = None
    ) -> Tensor:
        """
        Convert raw image to patch tokens with positional embeddings, CLS token, register tokens, and optional masks.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape `(B, C, H, W)`.
        masks : torch.Tensor, optional
            Boolean mask tensor of shape `(B, num_patches)`.

        Returns
        -------
        torch.Tensor
            Prepared token tensor of shape `(B, num_tokens + num_patches, D)`.
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        """
        Forward pass for a list of crop tensors and masks.

        Parameters
        ----------
        x_list : list of torch.Tensor
            List of image tensors.
        masks_list : list of torch.Tensor
            List of boolean mask tensors.

        Returns
        -------
        list of dict
            List of dictionaries containing normalized token representations per crop.
        """
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self, x: Union[Tensor, List[Tensor]], masks: Optional[Tensor] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Extract token features from a single image tensor or list of crops.

        Parameters
        ----------
        x : torch.Tensor or list of torch.Tensor
            Input image or crops.
        masks : torch.Tensor, optional
            Boolean mask tensor.

        Returns
        -------
        dict or list of dict
            Dictionary of token outputs.
        """
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: Union[int, Sequence] = 1
    ) -> List[Tensor]:
        """Extract intermediate features from non-chunked blocks."""
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(
        self, x: Tensor, n: Union[int, Sequence] = 1
    ) -> List[Tensor]:
        """Extract intermediate features from chunked FSDP blocks."""
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Retrieve intermediate block outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape `(B, C, H, W)`.
        n : int or sequence of int, default 1
            Number of last blocks to return, or sequence of block indices.
        reshape : bool, default False
            If True, reshapes spatial patch tokens to `(B, D, H_p, W_p)`.
        return_class_token : bool, default False
            If True, returns `(patch_tokens, cls_token)` tuples.
        norm : bool, default True
            Whether to apply final LayerNorm to intermediate outputs.

        Returns
        -------
        tuple
            Tuple of feature tensors or `(patch_tokens, cls_token)` pairs.
        """
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training: bool = False, **kwargs):
        """
        Forward pass returning token representations (training) or normalized CLS token (eval).

        Parameters
        ----------
        is_training : bool, default False
            If True, returns dictionary of all token outputs. If False, returns head(cls_token).

        Returns
        -------
        torch.Tensor or dict
            Evaluation representation or feature output dictionary.
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
