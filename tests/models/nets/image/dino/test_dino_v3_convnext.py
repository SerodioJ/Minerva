import pytest
import torch

from minerva.models.nets.image.dino.v3_convnext import (
    ConvNeXt,
    get_convnext_arch,
)


@pytest.fixture
def small_convnext():
    return ConvNeXt(
        in_chans=3,
        depths=[1, 1, 1, 1],
        dims=[32, 64, 128, 256],
        patch_size=16,
    )


def test_convnext_init(small_convnext):
    assert small_convnext.embed_dim == 256
    assert len(small_convnext.stages) == 4
    assert len(small_convnext.downsample_layers) == 4
    small_convnext.init_weights()


def test_convnext_forward_eval(small_convnext):
    small_convnext.eval()
    x = torch.randn(2, 3, 64, 64)
    out = small_convnext(x, is_training=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 256)


def test_convnext_forward_training(small_convnext):
    small_convnext.train()
    x = torch.randn(2, 3, 64, 64)
    out = small_convnext(x, is_training=True)
    assert isinstance(out, dict)
    assert "x_norm_clstoken" in out
    assert "x_storage_tokens" in out
    assert "x_norm_patchtokens" in out
    assert "x_prenorm" in out
    assert out["x_norm_clstoken"].shape == (2, 256)


def test_convnext_forward_features_list(small_convnext):
    x_list = [torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)]
    masks_list = [None, None]
    out_list = small_convnext.forward_features(x_list, masks=masks_list)
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    assert out_list[0]["x_norm_clstoken"].shape == (2, 256)


def test_convnext_get_intermediate_layers(small_convnext):
    x = torch.randn(2, 3, 64, 64)

    # Single last layer
    layers = small_convnext.get_intermediate_layers(x, n=1, reshape=False, norm=True)
    assert len(layers) == 1
    # Patch tokens: B x HW x C -> HW = (64 // 16) * (64 // 16) = 16
    assert layers[0].shape == (2, 16, 256)

    # Multiple layers with return_class_token and reshape
    layers_cls = small_convnext.get_intermediate_layers(
        x, n=2, reshape=True, return_class_token=True, norm=True
    )
    assert len(layers_cls) == 2
    for patch_feat, cls_token in layers_cls:
        assert patch_feat.ndim == 4
        assert cls_token.ndim == 2


def test_get_convnext_arch():
    for name in ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]:
        arch_fn = get_convnext_arch(name)
        assert callable(arch_fn)

    with pytest.raises(NotImplementedError):
        get_convnext_arch("convnext_unknown")
