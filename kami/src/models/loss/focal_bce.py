from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch
import torch.nn as nn

# https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html


class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float, gamma: float, weight: torch.Tensor = torch.tensor([1.0, 1.0])):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, masks):
        loss = self.weight[0] * self.bce_fn(logits[:, :, 0], labels[:, :, 0]) + self.weight[1] * sigmoid_focal_loss(
            logits[:, :, [1, 2]], labels[:, :, [1, 2]], alpha=self.alpha, gamma=self.gamma, reduction="mean"
        )
        return loss
