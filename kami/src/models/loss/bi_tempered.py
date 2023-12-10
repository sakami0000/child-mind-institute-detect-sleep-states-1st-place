#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-01-13 11:31
# @Author  : NingAnMe <ninganme@qq.com>
import torch
from torch.nn.modules.loss import _Loss


class BiTemperedLogisticLoss(_Loss):
    def __init__(
        self,
        reduction="mean",
        t1=1,
        t2=1,
        label_smoothing=0.0,
        num_iters=5,
    ):
        super().__init__(reduction=reduction)
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    @classmethod
    def log_t(cls, u, t):
        """Compute log_t for `u`."""

        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    @classmethod
    def exp_t(cls, u, t):
        """Compute exp_t for `u`."""

        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    @classmethod
    def compute_normalization_fixed_point(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu = torch.max(activations, dim=-1).values.unsqueeze(-1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < num_iters:
            i += 1
            logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).unsqueeze(-1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).unsqueeze(-1)

        return -cls.log_t(1.0 / logt_partition, t) + mu

    @classmethod
    def compute_normalization(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        if t < 1.0:
            return None  # not implemented as these values do not occur in the authors experiments...
        else:
            return cls.compute_normalization_fixed_point(activations, t, num_iters)

    @classmethod
    def tempered_softmax(cls, activations, t, num_iters=5):
        """Tempered softmax function.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """
        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
        else:
            normalization_constants = cls.compute_normalization(activations, t, num_iters)

        return cls.exp_t(activations - normalization_constants.unsqueeze(-1), t)

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets, masks):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        labels: A tensor with shape and dtype as activations.
        t1: Temperature 1 (< 1.0 for boundedness).
        t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
        label_smoothing: Label smoothing parameter between [0, 1).
        num_iters: Number of iterations to run the method.
        Returns:
        A loss tensor.
        """
        if self.label_smoothing > 0.0:
            targets = BiTemperedLogisticLoss._smooth_one_hot(targets, inputs.size(-1), self.label_smoothing)

        probabilities = self.tempered_softmax(inputs, self.t2, self.num_iters)

        temp1 = (self.log_t(targets + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * targets
        temp2 = (1 / (2 - self.t1)) * (torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss = temp1 - temp2

        loss = loss.sum(dim=-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
