import pytest
import torch

from minerva.models.nets.image.dino.v3_vit import (
    DinoVisionTransformer,
    cat_keep_shapes,
    uncat_with_shapes,
    vit_base,
    vit_giant2,
    vit_huge2,
    vit_large,
    vit_small,
    vit_so400m,
)


@pytest.fixture
def small_v3_vit():
    return DinoVisionTransformer(
        img_size=64,
        patch_size=16,
        in_chans=3,
        embed_dim=96,
        depth=2,
        num_heads=3,
        ffn_ratio=2.0,
        pos_embed_type="rope",
    )


def test_v3_vit_init(small_v3_vit):
    assert small_v3_vit.embed_dim == 96
    assert small_v3_vit.n_blocks == 2
    assert small_v3_vit.patch_embed.num_patches == (64 // 16) ** 2
    small_v3_vit.init_weights()


def test_v3_vit_forward_eval(small_v3_vit):
    small_v3_vit.eval()
    x = torch.randn(2, 3, 64, 64)
    out = small_v3_vit(x, is_training=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 96)


def test_v3_vit_forward_training(small_v3_vit):
    small_v3_vit.train()
    x = torch.randn(2, 3, 64, 64)
    out = small_v3_vit(x, is_training=True)
    assert isinstance(out, dict)
    assert "x_norm_clstoken" in out
    assert "x_norm_patchtokens" in out
    assert "x_prenorm" in out
    assert out["x_norm_clstoken"].shape == (2, 96)
    assert out["x_norm_patchtokens"].shape == (2, 16, 96)


def test_v3_vit_forward_with_masks(small_v3_vit):
    x = torch.randn(2, 3, 64, 64)
    masks = torch.zeros(2, 16, dtype=torch.bool)
    masks[:, :4] = True
    out = small_v3_vit.forward_features(x, masks=masks)
    assert "masks" in out
    assert out["x_norm_patchtokens"].shape == (2, 16, 96)


def test_v3_vit_forward_features_list(small_v3_vit):
    x_list = [torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)]
    masks_list = [None, None]
    out_list = small_v3_vit.forward_features(x_list, masks=masks_list)
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    assert out_list[0]["x_norm_clstoken"].shape == (2, 96)


def test_v3_vit_get_intermediate_layers(small_v3_vit):
    x = torch.randn(2, 3, 64, 64)

    # 1 last layer, not reshaped
    layers = small_v3_vit.get_intermediate_layers(x, n=1, reshape=False)
    assert len(layers) == 1
    assert layers[0].shape == (2, 16, 96)

    # 2 layers with class token and reshape
    layers_cls = small_v3_vit.get_intermediate_layers(
        x, n=2, reshape=True, return_class_token=True
    )
    assert len(layers_cls) == 2
    for patch_feat, cls_token in layers_cls:
        assert patch_feat.shape == (2, 96, 4, 4)
        assert cls_token.shape == (2, 96)


def test_cat_and_uncat_shapes():
    t1 = torch.randn(2, 5, 10)
    t2 = torch.randn(3, 4, 10)
    cat_t, shapes, num_tokens = cat_keep_shapes([t1, t2])
    assert cat_t.shape == (2 * 5 + 3 * 4, 10)

    uncatted = uncat_with_shapes(cat_t, shapes, num_tokens)
    assert len(uncatted) == 2
    assert uncatted[0].shape == t1.shape
    assert uncatted[1].shape == t2.shape
    assert torch.equal(uncatted[0], t1)
    assert torch.equal(uncatted[1], t2)


def test_v3_vit_factories():
    s = vit_small(patch_size=16, img_size=32, depth=1)
    assert isinstance(s, DinoVisionTransformer)
    assert s.embed_dim == 384

    b = vit_base(patch_size=16, img_size=32, depth=1)
    assert isinstance(b, DinoVisionTransformer)
    assert b.embed_dim == 768

    l = vit_large(patch_size=16, img_size=32, depth=1)
    assert isinstance(l, DinoVisionTransformer)
    assert l.embed_dim == 1024

    so = vit_so400m(patch_size=16, img_size=32, depth=1)
    assert isinstance(so, DinoVisionTransformer)
    assert so.embed_dim == 1152

    h = vit_huge2(patch_size=16, img_size=32, depth=1)
    assert isinstance(h, DinoVisionTransformer)
    assert h.embed_dim == 1280

    g = vit_giant2(patch_size=16, img_size=32, depth=1)
    assert isinstance(g, DinoVisionTransformer)
    assert g.embed_dim == 1536
