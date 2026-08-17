import pytest
import torch
import torch.nn as nn

from minerva.models.nets.image.dino.v2_vit import (
    DinoVisionTransformer,
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)


@pytest.fixture
def small_v2_vit():
    return DinoVisionTransformer(
        img_size=64,
        patch_size=16,
        in_chans=3,
        embed_dim=96,
        depth=2,
        num_heads=3,
        ffn_ratio=2.0,
        num_register_tokens=2,
    )


def test_v2_vit_init(small_v2_vit):
    assert small_v2_vit.embed_dim == 96
    assert small_v2_vit.n_blocks == 2
    assert small_v2_vit.num_register_tokens == 2
    assert small_v2_vit.cls_token is not None
    assert small_v2_vit.register_tokens is not None
    assert small_v2_vit.patch_embed.num_patches == (64 // 16) ** 2


def test_v2_vit_forward_eval(small_v2_vit):
    small_v2_vit.eval()
    x = torch.randn(2, 3, 64, 64)
    out = small_v2_vit(x, is_training=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 96)


def test_v2_vit_forward_training(small_v2_vit):
    small_v2_vit.train()
    x = torch.randn(2, 3, 64, 64)
    out = small_v2_vit(x, is_training=True)
    assert isinstance(out, dict)
    assert "x_norm_clstoken" in out
    assert "x_norm_regtokens" in out
    assert "x_norm_patchtokens" in out
    assert "x_prenorm" in out
    assert out["x_norm_clstoken"].shape == (2, 96)
    assert out["x_norm_regtokens"].shape == (2, 2, 96)
    assert out["x_norm_patchtokens"].shape == (2, 16, 96)


def test_v2_vit_forward_with_masks(small_v2_vit):
    x = torch.randn(2, 3, 64, 64)
    masks = torch.zeros(2, 16, dtype=torch.bool)
    masks[:, :4] = True
    out = small_v2_vit.forward_features(x, masks=masks)
    assert out["masks"] is not None
    assert out["x_norm_patchtokens"].shape == (2, 16, 96)


def test_v2_vit_forward_features_list(small_v2_vit):
    x_list = [torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)]
    masks_list = [
        torch.zeros(2, 16, dtype=torch.bool),
        torch.zeros(2, 16, dtype=torch.bool),
    ]
    out_list = small_v2_vit.forward_features(x_list, masks=masks_list)
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    assert out_list[0]["x_norm_clstoken"].shape == (2, 96)


def test_v2_vit_get_intermediate_layers(small_v2_vit):
    x = torch.randn(2, 3, 64, 64)

    # 1 last layer, not reshaped
    layers = small_v2_vit.get_intermediate_layers(x, n=1, reshape=False)
    assert len(layers) == 1
    assert layers[0].shape == (2, 16, 96)

    # 2 layers, reshaped and with class token
    layers_with_cls = small_v2_vit.get_intermediate_layers(
        x, n=2, reshape=True, return_class_token=True
    )
    assert len(layers_with_cls) == 2
    for patch_feat, cls_token in layers_with_cls:
        assert patch_feat.shape == (2, 96, 4, 4)
        assert cls_token.shape == (2, 96)


def test_v2_vit_swiglu_ffn():
    model = DinoVisionTransformer(
        img_size=32,
        patch_size=16,
        embed_dim=64,
        depth=1,
        num_heads=2,
        ffn_layer="swiglu",
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x, is_training=False)
    assert out.shape == (2, 64)


def test_v2_vit_factories():
    # Verify factory functions construct valid DinoVisionTransformer instances
    # with reduced layers/img_size for speed
    s = vit_small(patch_size=16, img_size=32, depth=1)
    assert isinstance(s, DinoVisionTransformer)
    assert s.embed_dim == 384

    b = vit_base(patch_size=16, img_size=32, depth=1)
    assert isinstance(b, DinoVisionTransformer)
    assert b.embed_dim == 768

    l = vit_large(patch_size=16, img_size=32, depth=1)
    assert isinstance(l, DinoVisionTransformer)
    assert l.embed_dim == 1024

    g = vit_giant2(patch_size=16, img_size=32, depth=1)
    assert isinstance(g, DinoVisionTransformer)
    assert g.embed_dim == 1536
