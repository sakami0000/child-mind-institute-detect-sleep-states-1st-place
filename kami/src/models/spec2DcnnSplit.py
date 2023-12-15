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
from src.models.loss.bi_tempered import BiTemperedLogisticLoss


class Spec2DCNNSplit(nn.Module):
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
        self.n_split = cfg.model.n_split
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
        elif self.cfg.loss.name == "bi_tempered":
            self.loss_fn = BiTemperedLogisticLoss(
                t1=self.cfg.loss.t1,
                t2=self.cfg.loss.t2,
                label_smoothing=self.cfg.loss.label_smoothing,
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

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)
        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        bach_size, height, n_timesteps = x.shape
        x_repeat = x.unsqueeze(dim=1).repeat(
            (1, self.n_split + 1, 1, 1)
        )  # (batch_size, n_split+1, height, n_timesteps)

        for si in range(self.n_split):
            n_group = 2 ** (si + 1)  # n_group分割
            len_group = n_timesteps // n_group
            for gi in range(0, n_group, 2):  # ２ペアずつ入れ替え処理
                start = len_group * gi
                mid = start + len_group
                end = start + len_group * 2
                temp = x_repeat[:, si + 1, :, start:mid].clone()
                x_repeat[:, si + 1, :, start:mid] = x_repeat[:, si + 1, :, mid:end]
                x_repeat[:, si + 1, :, mid:end] = temp
            # 現在値との差分
            x_repeat[:, si + 1, :, :] -= x_repeat[:, 0, :, :]
        x = x_repeat.reshape(bach_size, -1, n_timesteps)  # (batch_size, (n_split+1)* height, n_timesteps)

        # 最終的には 2**n_split 個ずつdecoderに入力する
        x_split = torch.split(x, x.shape[2] // (2**self.n_split), dim=2)
        x_list = []
        for x_gruop in x_split:
            x_list.append(self.decoder(x_gruop))
        logits = torch.cat(x_list, dim=1)  # (batch_size, n_timestep, n_classes)
        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels, masks)
            output["loss"] = loss

        return output
