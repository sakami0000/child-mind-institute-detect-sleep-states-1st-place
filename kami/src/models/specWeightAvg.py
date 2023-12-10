from typing import Optional
from omegaconf import DictConfig
from typing import Callable, Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup

from src.models.loss.tolerance import ToleranceLoss
from src.models.loss.tolerance_mse import ToleranceMSELoss
from src.models.loss.bce import BCEWithLogitsLoss
from src.models.loss.tolerance_nonzero import ToleranceNonZeroLoss
from src.models.loss.focal import FocalLoss
from src.models.loss.focal_bce import FocalBCELoss


# ref: https://github.com/analokmaus/kaggle-g2net-public/tree/main/models1d_pytorch
class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int | tuple = 128,
        kernel_sizes: tuple = (32, 16, 4, 2),
        stride: int = 4,
        sigmoid: bool = False,
        output_size: Optional[int] = None,
        conv: Callable = nn.Conv1d,
        reinit: bool = True,
    ):
        super().__init__()
        self.out_chans = len(kernel_sizes)
        self.out_size = output_size
        self.sigmoid = sigmoid
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])
        self.height = base_filters[-1]
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_block = [
                conv(
                    in_channels,
                    base_filters[0],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=(kernel_sizes[i] - 1) // 2,
                )
            ]
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_sizes[i],
                            stride=stride,
                            padding=(kernel_sizes[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((self.out_size, None))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (_type_): (batch_size, in_channels, time_steps)

        Returns:
            _type_: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)

        for i in range(self.out_chans):
            x = self.spec_conv[i](x)  # (batch_size, base_filters[-1], time_steps)
        if self.out_size is not None:
            x = self.pool(x)  # (batch_size, height, out_size)
        if self.sigmoid:
            x = x.sigmoid()
        return x


class SpecWeightAvg(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        feature_extractor: nn.Module,
        decoders: list[nn.Module],
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

        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.decoder_weight = decoders[0]
        self.decoder_sleep = decoders[1]
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_weight = torch.tensor(cfg.loss.loss_weight) if "loss_weight" in cfg.loss else None
        self.label_weight = torch.tensor(cfg.label_weight) if "label_weight" in cfg else None
        self.pos_weight = torch.tensor(cfg.pos_weight) if "pos_weight" in cfg else None
        self.loss_fn = None

        base_filters = 64
        kernel_sizes = [180, 30, 6]
        stride = 1
        """
        self.cnn_weight = CNN(
            in_channels=height,
            base_filters=base_filters,
            kernel_sizes=kernel_sizes,
            stride=stride,
            sigmoid=False,
            output_size=1,
            conv=nn.Conv1d,
            reinit=True,
        )
        self.cnn_sleep = CNN(
            in_channels=height,
            base_filters=base_filters,
            kernel_sizes=kernel_sizes,
            stride=stride,
            sigmoid=False,
            output_size=1,
            conv=nn.Conv1d,
            reinit=True,
        )
        """

        # 0から1までのなだらかな直線を作る
        filter_size = cfg.model.filter_size
        # 左で起きていたが、右で寝ている場合
        self.onset_filter_weight = (
            (
                torch.cat([torch.linspace(0, -1, filter_size // 2), torch.linspace(1, 0, filter_size // 2)], dim=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            .type(torch.cuda.HalfTensor)
            .requires_grad_(False)
        )
        # 左で寝ているが、右で起きている場合
        self.wakeup_filter_weight = (
            (
                torch.cat([torch.linspace(0, 1, filter_size // 2), torch.linspace(-1, 1, filter_size // 2)], dim=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            .type(torch.cuda.HalfTensor)
            .requires_grad_(False)
        )

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

        x = self.encoder(x)  # (batch_size, 1, height, n_timesteps)
        x = x.squeeze(1)  # (batch_size, height, n_timesteps)

        x_weight = self.decoder_weight(x)  # (batch_size, n_timesteps, 1)
        x_sleep = self.decoder_sleep(x)  # (batch_size, n_timesteps, 1)
        x_sleep_weighted = x_sleep * x_weight  # (batch_size, n_timesteps, 1)

        x_onset = F.conv1d(x_sleep_weighted.squeeze(2).unsqueeze(1), self.onset_filter_weight, padding="same")
        x_wakeup = F.conv1d(x_sleep_weighted.squeeze(2).unsqueeze(1), self.wakeup_filter_weight, padding="same")

        logits = torch.stack(
            [x_sleep.squeeze(2), x_onset.squeeze(1), x_wakeup.squeeze(1)], dim=2
        )  # (batch_size, n_timesteps, 3)

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels, masks)
            output["loss"] = loss

        return output
