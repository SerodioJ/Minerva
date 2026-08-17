import pytest
import torch
import torch.nn.functional as F

from minerva.losses.dino import (
    DINOLoss,
    GramLoss,
    KoLeoLoss,
    KoLeoLossDistributed,
    SinkhornKnoppTeacher,
    iBOTPatchLoss,
    lossfunc,
)


# -------------------------------------------------------------
# GramLoss tests
# -------------------------------------------------------------
def test_gram_loss_image_level():
    criterion = GramLoss(apply_norm=True, remove_neg=True)
    out_feats = torch.randn(4, 16, 64)
    target_feats = torch.randn(4, 16, 64)
    loss = criterion(out_feats, target_feats, img_level=True)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_gram_loss_batch_level():
    criterion = GramLoss(
        apply_norm=False, remove_neg=False, remove_only_teacher_neg=True
    )
    out_feats = torch.randn(4, 16, 32)
    target_feats = torch.randn(4, 16, 32)
    loss = criterion(out_feats, target_feats, img_level=False)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gram_loss_identical_inputs():
    criterion = GramLoss(apply_norm=True, remove_neg=True)
    feats = torch.randn(2, 8, 16)
    loss = criterion(feats, feats, img_level=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# -------------------------------------------------------------
# SinkhornKnopp & helper tests
# -------------------------------------------------------------
def test_lossfunc():
    t = F.softmax(torch.randn(4, 10), dim=-1)
    s = torch.randn(4, 10)
    loss = lossfunc(t, s, temp=0.1)
    assert loss.shape == (4,)
    assert torch.all(torch.isfinite(loss))


def test_sinkhorn_knopp_teacher():
    sk = SinkhornKnoppTeacher()
    teacher_output = torch.randn(20, 10)
    n_masked = torch.tensor(20)
    Q = sk(
        teacher_output,
        teacher_temp=0.1,
        n_masked_patches_tensor=n_masked,
        n_iterations=3,
    )
    assert Q.shape == (20, 10)
    assert torch.all(Q >= 0)
    # Total sum should be approximately n_masked
    assert Q.sum().item() == pytest.approx(20.0, rel=1e-2)


# -------------------------------------------------------------
# iBOTPatchLoss tests
# -------------------------------------------------------------
def test_ibot_patch_loss_forward():
    criterion = iBOTPatchLoss(patch_out_dim=64, student_temp=0.1)
    criterion.init_weights()

    B, N, D = 2, 8, 64
    student_tokens = torch.randn(B, N, D)
    teacher_tokens = F.softmax(torch.randn(B, N, D), dim=-1)
    masks = torch.ones(B, N, dtype=torch.bool)

    loss = criterion(student_tokens, teacher_tokens, masks)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ibot_patch_loss_forward_masked():
    criterion = iBOTPatchLoss(patch_out_dim=32, student_temp=0.1)
    criterion.init_weights()

    total_masked = 10
    D = 32
    s_masked = torch.randn(total_masked, D)
    t_masked = F.softmax(torch.randn(total_masked, D), dim=-1)
    masks_flat = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )

    loss = criterion.forward_masked(
        s_masked,
        t_masked,
        masks_flat,
        n_masked_patches=7,
    )
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


def test_ibot_patch_loss_center_update():
    criterion = iBOTPatchLoss(patch_out_dim=16, center_momentum=0.9)
    criterion.init_weights()

    teacher_tokens = torch.ones(2, 4, 16) * 2.0
    criterion.update_center(teacher_tokens)
    assert criterion.updated is False
    criterion.apply_center_update()
    assert criterion.updated is True
    assert not torch.all(torch.isnan(criterion.center))


# -------------------------------------------------------------
# KoLeoLoss tests
# -------------------------------------------------------------
def test_koleo_loss_forward():
    criterion = KoLeoLoss()
    student_output = torch.randn(8, 32)
    loss = criterion(student_output)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_koleo_loss_distributed_single_rank():
    criterion = KoLeoLossDistributed(topk=1, loss_group_size=4)
    student_output = torch.randn(4, 16)
    loss = criterion(student_output)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_koleo_loss_distributed_invalid_group_size():
    criterion = KoLeoLossDistributed(topk=1, loss_group_size=3)
    student_output = torch.randn(4, 16)
    with pytest.raises(ValueError):
        criterion(student_output)


# -------------------------------------------------------------
# DINOLoss tests
# -------------------------------------------------------------
def test_dino_loss_v2():
    criterion = DINOLoss(out_dim=32, student_temp=0.1, dino_version=2)
    criterion.init_weights()

    s_out = [torch.randn(4, 32), torch.randn(4, 32)]
    t_out = [
        F.softmax(torch.randn(4, 32), dim=-1),
        F.softmax(torch.randn(4, 32), dim=-1),
    ]

    loss = criterion(
        student_output_list=s_out,
        teacher_out_softmaxed_centered_list=t_out,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dino_loss_v3():
    criterion = DINOLoss(out_dim=32, student_temp=0.1, dino_version=3)
    criterion.init_weights()

    # student_logits: [student_crops, B, K]
    student_logits = torch.randn(6, 4, 32)
    # teacher_probs: [teacher_crops, B, K]
    teacher_probs = F.softmax(torch.randn(2, 4, 32), dim=-1)

    loss = criterion(
        student_logits=student_logits,
        teacher_probs=teacher_probs,
        ignore_diagonal=False,
    )
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)

    loss_diag = criterion(
        student_logits=student_logits,
        teacher_probs=teacher_probs,
        ignore_diagonal=True,
    )
    assert isinstance(loss_diag, torch.Tensor)
    assert torch.isfinite(loss_diag)


def test_dino_loss_center_update():
    criterion = DINOLoss(out_dim=16, center_momentum=0.9, dino_version=3)
    criterion.init_weights()

    teacher_output = torch.randn(4, 16)
    criterion.update_center(teacher_output)
    assert criterion.updated is False
    criterion.apply_center_update()
    assert criterion.updated is True
    assert criterion.center.shape == (1, 16)


def test_dino_loss_invalid_version():
    with pytest.raises(ValueError):
        criterion = DINOLoss(out_dim=16, dino_version=99)
        criterion(
            student_logits=torch.randn(2, 2, 16), teacher_probs=torch.randn(2, 2, 16)
        )
