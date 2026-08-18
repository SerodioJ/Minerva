# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# References:
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/koleo_loss.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/ibot_patch_loss.py
#   https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/dino_clstoken_loss.py

import math

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """
    Gram matrix matching loss between student and teacher patch feature representations.

    Parameters
    ----------
    apply_norm : bool, default True
        Whether to L2-normalize patch features before computing the Gram matrix.
    img_level : bool, default True
        If True, computes Gram matrix per image. If False, computes across the full batch.
    remove_neg : bool, default True
        If True, zeros out negative correlation values in both student and teacher Gram matrices.
    remove_only_teacher_neg : bool, default False
        If True, zeros out only the negative correlation values in the teacher Gram matrix.
    """

    def __init__(
        self,
        apply_norm=True,
        img_level=True,
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()

        # Loss
        self.mse_loss = torch.nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):
        """
        Compute the MSE loss between Gram matrices of student and teacher features.

        Parameters
        ----------
        output_feats : torch.Tensor
            Student feature tensor of shape `(B, N, D)` or `(B*N, D)` if `img_level=False`.
        target_feats : torch.Tensor
            Teacher feature tensor of shape `(B, N, D)` or `(B*N, D)` if `img_level=False`.
        img_level : bool, default True
            If True, computes Gram matrices per-image; otherwise computes across the flattened batch.

        Returns
        -------
        torch.Tensor
            Scalar MSE loss value.
        """

        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        # Float casting
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # SSL correlation
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            target_feats = target_feats.flatten(0, 1)

        # Compute similarities
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # Patch correlation
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            output_feats = output_feats.flatten(0, 1)

        # Compute similarities
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            target_sim[target_sim < 0] = 0.0
            # student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)


def lossfunc(
    t: torch.Tensor, s: torch.Tensor, temp: float
) -> torch.Tensor:  # noqa: F811
    """
    Cross-entropy loss component between teacher probabilities and student logits.

    Parameters
    ----------
    t : torch.Tensor
        Teacher target probabilities.
    s : torch.Tensor
        Student unnormalized logits.
    temp : float
        Student temperature parameter.

    Returns
    -------
    torch.Tensor
        Cross-entropy values summed across the class/prototype dimension.
    """
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class SinkhornKnoppTeacher(nn.Module):
    """
    Sinkhorn-Knopp algorithm module for teacher prototype assignment normalization.

    NOTE: This is implemented as an nn.Module rather than a standalone function
    to allow module-level compilation with `torch.compile`.
    """

    @torch.no_grad()
    def forward(
        self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3
    ):
        """
        Run Sinkhorn-Knopp normalization on teacher outputs.

        Parameters
        ----------
        teacher_output : torch.Tensor
            Teacher unnormalized logits.
        teacher_temp : float
            Teacher temperature scaling.
        n_masked_patches_tensor : torch.Tensor
            Tensor holding the total number of masked patches.
        n_iterations : int, default 3
            Number of Sinkhorn-Knopp normalization iterations.

        Returns
        -------
        torch.Tensor
            Normalized soft assignment distribution matrix.
        """
        teacher_output = teacher_output.float()
        # world_size = torch_dist.get_world_size() if torch_dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor
        if torch_dist.is_initialized():
            torch_dist.all_reduce(
                B, group=None
            )  # TODO change to local reduce if necessary
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch_dist.is_initialized():
            torch_dist.all_reduce(
                sum_Q, group=None
            )  # TODO change to local reduce if necessary
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if torch_dist.is_initialized():
                torch_dist.all_reduce(
                    sum_of_rows, group=None
                )  # TODO change to local reduce if necessary
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class iBOTPatchLoss(nn.Module):
    """
    Masked image modeling patch loss (iBOT) for DINO self-supervised learning.

    Parameters
    ----------
    patch_out_dim : int
        Output dimensionality of the patch projection head (number of patch prototypes).
    student_temp : float, default 0.1
        Temperature parameter for the student network softmax.
    center_momentum : float, default 0.9
        Momentum rate used for exponential moving average updates of the teacher center.
    """

    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, 1, patch_out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None
        self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher()
        self.sinkhorn_knopp_teacher.compile()

    def init_weights(self) -> None:
        """Initialize teacher center buffer with zeros."""
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_patch_tokens, teacher_temp, update_centers=True
    ):
        """
        Apply centering and softmax temperature scaling to teacher patch tokens.

        Parameters
        ----------
        teacher_patch_tokens : torch.Tensor
            Unnormalized teacher patch token logits.
        teacher_temp : float
            Teacher sharpening temperature.
        update_centers : bool, default True
            Whether to apply any pending asynchronous center updates.

        Returns
        -------
        torch.Tensor
            Softmax probabilities for teacher patch tokens.
        """
        if update_centers:
            self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Compute cross-entropy loss between softmax outputs of student and teacher on masked patches.

        Parameters
        ----------
        student_patch_tokens : torch.Tensor
            Student patch token predictions of shape `(B, N, D)`.
        teacher_patch_tokens : torch.Tensor
            Teacher patch token probabilities of shape `(B, N, D)`.
        student_masks_flat : torch.Tensor
            Boolean mask tensor of shape `(B, N)` indicating masked patches.

        Returns
        -------
        torch.Tensor
            Mean patch cross-entropy loss.
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = lossfunc(t, s, self.student_temp)
        loss = torch.sum(
            loss * student_masks_flat.float(), dim=-1
        ) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        """
        Compute masked patch loss directly on already-indexed masked tokens.

        Parameters
        ----------
        student_patch_tokens_masked : torch.Tensor
            Student patch logits for masked positions.
        teacher_patch_tokens_masked : torch.Tensor
            Teacher target probabilities for masked positions.
        student_masks_flat : torch.Tensor
            Boolean mask tensor indicating masked positions.
        n_masked_patches : int, optional
            Number of valid masked patches to truncate loss to.
        masks_weight : torch.Tensor, optional
            Optional weighting tensor per masked patch.

        Returns
        -------
        torch.Tensor
            Scalar loss value normalized by batch size.
        """
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        """Start asynchronous reduction to update teacher center buffer."""
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        """
        Compute batch center and initiate asynchronous all-reduce across distributed ranks.

        Parameters
        ----------
        teacher_patch_tokens : torch.Tensor
            Teacher patch tokens from the current batch.
        """
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(
            teacher_patch_tokens.mean(1), dim=0, keepdim=True
        )
        if torch_dist.is_initialized():
            self.reduce_handle = torch_dist.all_reduce(
                self.async_batch_center, async_op=True, group=None
            )  # TODO change to local reduce if necessary

    @torch.no_grad()
    def apply_center_update(self):
        """Finalize asynchronous center reduction and apply exponential moving average update."""
        if self.updated is False:
            world_size = (
                torch_dist.get_world_size() if torch_dist.is_initialized() else 1
            )

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic loss regularizer to maximize representation uniformity on the unit sphere.

    References
    ----------
    Sablayrolles et al., "Spreading vectors for similarity search", 2018.
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors via GPU inner product.

        Parameters
        ----------
        x : torch.Tensor
            L2-normalized feature vectors of shape `(B, D)`.

        Returns
        -------
        torch.Tensor
            Indices of the nearest neighbor for each sample.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        _, indices = torch.max(dots, dim=1)  # max inner prod -> min distance
        return indices

    def forward(self, student_output, eps=1e-8):
        """
        Compute KoLeo entropy regularization loss.

        Parameters
        ----------
        student_output : torch.Tensor
            Student backbone output representations of shape `(B, D)`.
        eps : float, default 1e-8
            Small epsilon for numerical stability.

        Returns
        -------
        torch.Tensor
            KoLeo loss scalar.
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(
                student_output, student_output[indices]
            )  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


class KoLeoLossDistributed(nn.Module):
    """
    Distributed Kozachenko-Leonenko entropic loss regularizer computed across multiple GPUs.

    Parameters
    ----------
    topk : int, default 1
        Number of nearest neighbors to consider.
    loss_group_size : int, optional
        Size of nearest neighbor feature candidate set. If None, uses global batch size.
    """

    def __init__(self, topk=1, loss_group_size: int | None = None):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.topk = topk
        self.loss_group_size = loss_group_size  # Size of the nearest neighbor set. If None, uses global batch size.

    def pairwise_NNs_inner(self, x, all_x, rank):
        """
        Find nearest neighbors in the global feature set for local batch vectors.

        Parameters
        ----------
        x : torch.Tensor
            Local feature representations of shape `(local_B, D)`.
        all_x : torch.Tensor
            Gathered feature representations of shape `(global_B, D)`.
        rank : int
            Rank index within the loss group.

        Returns
        -------
        torch.Tensor
            Indices of top-k nearest neighbors in `all_x`.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, all_x.t())  # local_B x global_B
        local_B, global_B = dots.shape
        dots.view(-1)[rank * local_B :: (global_B + 1)].fill_(
            -1
        )  # Trick to fill diagonal with -1
        _, indices = torch.topk(
            dots, dim=1, k=self.topk
        )  # max inner prod -> min distance
        return indices

    def forward(self, student_output, eps=1e-8):
        """
        Compute distributed KoLeo loss across GPUs.

        Parameters
        ----------
        student_output : torch.Tensor
            Local student backbone output representations of shape `(local_B, D)`.
        eps : float, default 1e-8
            Small epsilon for numerical stability.

        Returns
        -------
        torch.Tensor
            Distributed KoLeo loss scalar.
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(
                student_output, eps=eps, p=2, dim=-1
            )  # local_B x D

            if torch_dist.is_initialized():
                all_student_outputs = torch.cat(
                    torch_dist.nn.all_gather(student_output), dim=0
                )  # global_B x D
                world_size = torch_dist.get_world_size()
                rank = torch_dist.get_rank()
            else:
                all_student_outputs = student_output
                world_size = 1
                rank = 0

            # Group the global batch into groups of size `loss_group_size` and use the features of the group
            # the local rank falls into as the nearest neighbor set for the local rank
            local_B = len(student_output)
            global_B = len(all_student_outputs)
            loss_group_size = (
                self.loss_group_size if self.loss_group_size is not None else global_B
            )
            if loss_group_size % local_B != 0:
                raise ValueError(
                    f"Loss group size size {loss_group_size} must be a multiple of local batch size {local_B}."
                )
            if global_B % loss_group_size != 0:
                raise ValueError(
                    f"Global batch size {global_B} must be divisible by loss group size {loss_group_size}."
                )
            n_groups = global_B // loss_group_size
            ranks_per_group = world_size // n_groups
            rank_in_group = rank % ranks_per_group
            group = rank // ranks_per_group
            all_student_outputs = all_student_outputs.view(
                n_groups, loss_group_size, student_output.shape[1]
            )
            all_student_outputs = all_student_outputs[group]  # loss_group_size x D

            with torch.no_grad():
                indices = self.pairwise_NNs_inner(
                    student_output, all_student_outputs, rank_in_group
                )  # local_B x topk

            student_output_expanded = (
                student_output.unsqueeze(1).repeat(1, self.topk, 1).flatten(0, 1)
            )  # (local_B * topk) x D
            distances = self.pdist(
                student_output_expanded, all_student_outputs[indices].flatten(0, 1)
            )  # BxD, BxD -> B
            loss = -torch.log(distances.float() + eps).mean()

        return loss


class DINOLoss(nn.Module):
    """
    DINO loss module supporting both DINOv2 and DINOv3 formulations with centering and sharpening.

    Parameters
    ----------
    out_dim : int
        Number of output prototypes for the DINO projection head.
    student_temp : float, default 0.1
        Temperature for the student network softmax.
    center_momentum : float, default 0.9
        Momentum rate for the teacher center buffer EMA update.
    dino_version : {2, 3}, default 3
        DINO version formulation to use for forward loss calculation.
    """

    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
        dino_version=3,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.dino_version = dino_version

    def init_weights(self) -> None:
        """Initialize teacher center buffer with zeros."""
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        """
        Center and sharpen teacher outputs using softmax.

        Parameters
        ----------
        teacher_output : torch.Tensor
            Teacher unnormalized logits.
        teacher_temp : float
            Teacher sharpening temperature.
        update_centers : bool, default True
            Whether to apply pending asynchronous center updates.

        Returns
        -------
        torch.Tensor
            Sharpened teacher probabilities.
        """
        if update_centers:
            self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """
        Run Sinkhorn-Knopp algorithm on teacher predictions.

        Parameters
        ----------
        teacher_output : torch.Tensor
            Teacher logits of shape `(batch, prototypes)`.
        teacher_temp : float
            Teacher sharpening temperature.
        n_iterations : int, default 3
            Number of normalization iterations.

        Returns
        -------
        torch.Tensor
            Target distribution matrix Q.
        """
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()
        world_size = torch_dist.get_world_size() if torch_dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch_dist.is_initialized():
            torch_dist.all_reduce(
                sum_Q, group=None
            )  # TODO change to local reduce if necessary
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if torch_dist.is_initialized():
                torch_dist.all_reduce(
                    sum_of_rows, group=None
                )  # TODO change to local reduce if necessary
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, **kwargs):
        """
        Dispatch forward pass based on `self.dino_version`.

        Returns
        -------
        torch.Tensor
            Computed DINO loss scalar.
        """
        if self.dino_version == 2:
            return self.v2_forward(**kwargs)
        elif self.dino_version == 3:
            return self.v3_forward(**kwargs)
        else:
            raise ValueError(f"Unknown DINO version: {self.dino_version}. Use 2 or 3.")

    def v2_forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Compute DINOv2 multi-crop cross-entropy loss between student outputs and teacher probabilities.

        Parameters
        ----------
        student_output_list : list of torch.Tensor
            List of student crop logits.
        teacher_out_softmaxed_centered_list : list of torch.Tensor
            List of centered and softmaxed teacher crop probabilities.

        Returns
        -------
        torch.Tensor
            Total DINOv2 loss across all crop pairs.
        """
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    def v3_forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """
        Compute DINOv3 vectorized cross-entropy loss between student logits and teacher probabilities.

        Parameters
        ----------
        student_logits : torch.Tensor
            Student predictions of shape `(student_crops, B, K)`.
        teacher_probs : torch.Tensor
            Teacher probability targets of shape `(teacher_crops, B, K)`.
        ignore_diagonal : bool, default False
            Whether to exclude matched crop index pairs `s == t` from loss.

        Returns
        -------
        torch.Tensor
            Mean DINOv3 cross-entropy loss.
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = F.log_softmax(
            student_logits.float() / self.student_temp, dim=-1
        )
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Start asynchronous reduction to update teacher center buffer."""
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        """
        Compute batch center and initiate asynchronous all-reduce.

        Parameters
        ----------
        teacher_output : torch.Tensor
            Teacher prototype logits for the current batch.
        """
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if torch_dist.is_initialized():
            self.reduce_handle = torch_dist.all_reduce(
                self.async_batch_center, async_op=True, group=None
            )  # TODO change to local reduce if necessary

    @torch.no_grad()
    def apply_center_update(self):
        """Finalize asynchronous center reduction and apply exponential moving average update."""
        if self.updated is False:
            world_size = (
                torch_dist.get_world_size() if torch_dist.is_initialized() else 1
            )

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True
