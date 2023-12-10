from typing import Optional
from omegaconf import DictConfig

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup

from src.models.loss.tolerance import ToleranceLoss
from src.models.loss.tolerance_mse import ToleranceMSELoss
from src.models.loss.bce import BCEWithLogitsLoss
from src.models.loss.tolerance_nonzero import ToleranceNonZeroLoss
from src.models.loss.focal import FocalLoss
from src.models.loss.focal_bce import FocalBCELoss


class Spec2DCNNMinMax(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        height: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = feature_extractor
        aux_params = dict(
            pooling="avg",  # one of 'avg', 'max'
            dropout=0.1,  # dropout ratio, default is None
            activation=None,  # activation function, default is None
            classes=2 * height,  # define number of output labels
        )

        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            aux_params=aux_params,
        )
        self.decoder = decoder
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_weight = torch.tensor(cfg.loss.loss_weight) if "loss_weight" in cfg.loss else None
        self.label_weight = torch.tensor(cfg.label_weight) if "label_weight" in cfg else None
        self.pos_weight = torch.tensor(cfg.pos_weight) if "pos_weight" in cfg else None
        self.loss_fn = None
        self.update_loss_fn()

    def update_loss_fn(self, sleep_decay: float = 1.0) -> None:
        self.label_weight[0] = self.label_weight[0] * sleep_decay
        if self.cfg.loss.name == "tolerance":
            self.loss_fn = ToleranceLoss(
                loss_weight=self.loss_weight, label_weight=self.label_weight, pos_weight=self.pos_weight
            )
        elif self.cfg.loss.name == "tolerance_mse":
            self.loss_fn = ToleranceMSELoss(
                loss_weight=self.loss_weight, label_weight=self.label_weight, pos_weight=self.pos_weight
            )
        elif self.cfg.loss.name == "tolerance_nonzero":
            self.loss_fn = ToleranceNonZeroLoss(
                loss_weight=self.loss_weight, label_weight=self.label_weight, pos_weight=self.pos_weight
            )
        elif self.cfg.loss.name == "focal":
            self.loss_fn = FocalLoss(
                alpha=self.cfg.loss.alpha,
                gamma=self.cfg.loss.gamma,
            )
        elif self.cfg.loss.name == "focal_bce":
            self.loss_fn = FocalBCELoss(
                alpha=self.cfg.loss.alpha,
                gamma=self.cfg.loss.gamma,
                weight=torch.tensor(self.cfg.loss.weight),
            )
        else:
            self.loss_fn = BCEWithLogitsLoss(weight=self.label_weight, pos_weight=self.pos_weight)

        self.loss_fn = self.loss_fn.cuda()

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

        x, mid = self.encoder(x)  # (batch_size, 1, height, n_timesteps), (batch_size, 2 * height)
        x_min, x_max = torch.split(mid, mid.shape[1] // 2, dim=1)
        x = x.squeeze(1)  # (batch_size, height, n_timesteps)
        x = torch.cat(
            [x, x - x_min.unsqueeze(2), x_max.unsqueeze(2) - x], dim=1
        )  # (batch_size, 3 * height, n_timesteps)
        logits = self.decoder(x)  # (batch_size, n_classes, n_timesteps)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels, masks)
            output["loss"] = loss

        return output
