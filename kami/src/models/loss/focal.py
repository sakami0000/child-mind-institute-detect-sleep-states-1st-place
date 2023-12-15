from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch
import torch.nn as nn

# https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        gamma: float,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels, masks):
        loss = sigmoid_focal_loss(
            logits[:, :, [1, 2]], labels[:, :, [1, 2]], alpha=self.alpha, gamma=self.gamma, reduction="mean"
        )
        return loss
