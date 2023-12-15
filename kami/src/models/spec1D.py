from typing import Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup

from src.models.loss.tolerance import ToleranceLoss
from src.models.loss.tolerance_mse import ToleranceMSELoss
from src.models.loss.bce import BCEWithLogitsLoss
from src.models.loss.tolerance_nonzero import ToleranceNonZeroLoss


class Spec1D(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.channels_fc = nn.Linear(feature_extractor.out_chans, 1)
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_weight = torch.tensor(cfg.loss.loss_weight) if "loss_weight" in cfg.loss else None
        self.label_weight = torch.tensor(cfg.label_weight) if "label_weight" in cfg else None
        self.pos_weight = torch.tensor(cfg.pos_weight) if "pos_weight" in cfg else None

        if cfg.loss.name == "tolerance":
            self.loss_fn = ToleranceLoss(
                loss_weight=self.loss_weight, label_weight=self.label_weight, pos_weight=self.pos_weight
            )
        elif cfg.loss.name == "tolerance_mse":
            self.loss_fn = ToleranceMSELoss(
                loss_weight=self.loss_weight, label_weight=self.label_weight, pos_weight=self.pos_weight
            )
        elif cfg.loss.name == "tolerance_nonzero":
            self.loss_fn = ToleranceNonZeroLoss(
                loss_weight=self.loss_weight, label_weight=self.label_weight, pos_weight=self.pos_weight
            )
        else:
            self.loss_fn = BCEWithLogitsLoss(weight=self.label_weight, pos_weight=self.pos_weight)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        # pool over n_channels dimension
        x = x.transpose(1, 3)  # (batch_size, n_timesteps, height, n_channels)
        x = self.channels_fc(x)  # (batch_size, n_timesteps, height, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (batch_size, height, n_timesteps)
        logits = self.decoder(x)  # (batch_size, n_classes, n_timesteps)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels, masks)
            output["loss"] = loss

        return output
