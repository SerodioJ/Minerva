# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# References:
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/layers/dino_head.py

import os
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.nn.init import trunc_normal_, orthogonal_


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        dino_version=3,
    ):
        super().__init__()
        self.dino_version = dino_version
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        if self.dino_version == 2:
            self.apply(self._init_weights)
            self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

            # Create the 'g' scale parameter explicitly (initialized to 1.0)
            # Shape must broadcast correctly with the weight tensor (out_dim, in_dim)
            self.weight_g = nn.Parameter(torch.ones(out_dim, 1))
        elif self.dino_version == 3:
            self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        else:
            raise ValueError(f"Unknown DINO version: {self.dino_version}. Use 2 or 3.")

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # orthogonal_(m.weight) # orthogonal init test
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            if getattr(self, "dino_version", 3) == 2:
                weight_v = self.last_layer.weight
                weight_v_norm = nn.functional.normalize(weight_v, dim=1, p=2)

                # Scale the normalized weight by our learnable parameter 'g'
                normed_weight = self.weight_g * weight_v_norm
                x = nn.functional.linear(x, normed_weight)
            else:
                x = self.last_layer(x)
        return x


def _build_mlp(
    nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True
):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


def get_activation_checkpoint_wrapper(checkpointing_full: bool):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    if checkpointing_full:
        _checkpointing_wrapper = checkpoint_wrapper
    else:
        _save_list = [
            # mm
            torch.ops.aten.mm.default,
            torch.ops.aten._scaled_mm.default,
            # attentions
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
        ]
        _checkpointing_wrapper = partial(
            checkpoint_wrapper,
            context_fn=partial(create_selective_checkpoint_contexts, _save_list),
            preserve_rng_state=True,
        )
    return _checkpointing_wrapper


def wrap_compile_block(
    module: nn.Module, use_cuda_graphs: bool, is_backbone_block: bool
) -> nn.Module:
    if use_cuda_graphs and is_backbone_block:
        module.compile(
            fullgraph=True, dynamic=False, options={"triton.cudagraphs": True}
        )
    else:
        module.compile()
    return module
