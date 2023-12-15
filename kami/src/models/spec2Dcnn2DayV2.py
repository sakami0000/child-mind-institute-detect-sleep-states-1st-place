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


class Spec2DCNN2DayV2(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = feature_extractor
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
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
            x (torch.Tensor): (batch_size, n_channels, duration)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        x1, x2 = torch.split(x, x.shape[2] // 2, dim=2)

        # 2日分のデータを分割して並行に入力
        x1 = self.feature_extractor(x1)  # (batch_size, n_channels, height, n_timesteps//2)
        x2 = self.feature_extractor(x2)  # (batch_size, n_channels, height, n_timesteps//2)

        x1 = self.encoder(x1).squeeze(1)  # (batch_size, height, n_timesteps//2)
        x2 = self.encoder(x2).squeeze(1)  # (batch_size, height, n_timesteps//2)

        # 残差を結合してデコーダーに入力
        x = torch.cat([x1, x1 - x2], dim=1)  # (batch_size, n_classes, n_timesteps//2)
        logits1 = self.decoder(x)  # (batch_size, n_timesteps//2, n_classes)

        x = torch.cat([x2, x2 - x1], dim=1)
        logits2 = self.decoder(x)

        logits = torch.cat([logits1, logits2], dim=1)  # (batch_size, n_timesteps, n_classes)
        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels, masks)
            output["loss"] = loss

        return output
