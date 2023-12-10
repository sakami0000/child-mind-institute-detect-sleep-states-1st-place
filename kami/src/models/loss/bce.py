import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, pos_weight: torch.Tensor = None):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(weight=self.weight, pos_weight=self.pos_weight)

    def forward(self, logits, labels, masks):
        loss = self.loss_fn(logits, labels)
        return loss
