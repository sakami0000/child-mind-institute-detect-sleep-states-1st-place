import torch
import torch.nn as nn


class ToleranceLoss(nn.Module):
    def __init__(
        self,
        loss_weight: torch.Tensor = torch.Tensor([0.5, 0.5]),
        label_weight: torch.Tensor = None,
        pos_weight: torch.Tensor = None,
    ):
        super(ToleranceLoss, self).__init__()

        self.loss_weight = loss_weight
        self.label_weight = label_weight
        self.pos_weight = pos_weight

        self.loss_fn = nn.BCEWithLogitsLoss(weight=self.label_weight, pos_weight=self.pos_weight)

    def forward(self, logits, labels, masks):
        # logits: shape (batch, seq, class)
        # masks: shape (batch, tolerance, seq, class)

        logits_event = logits[:, :, [1, 2]]

        # Step 1: Find the maximum value of logits in the range of each mask
        masked_logits_max, _ = torch.max(logits_event.unsqueeze(1) * masks, dim=2)  # (batch, tolerance, class)

        # Step 2: Calculate max(0, logits - x) and take the average over seq
        loss = torch.relu(logits_event.unsqueeze(1) - masked_logits_max.unsqueeze(2))  # (batch, tolerance, seq, class)
        loss = torch.mean(loss)

        loss = self.loss_weight[0] * self.loss_fn(logits, labels) + self.loss_weight[1] * loss
        return loss
