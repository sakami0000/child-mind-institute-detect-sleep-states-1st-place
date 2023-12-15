from typing import Optional
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchvision.transforms.functional import resize
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.datamodule.seg import nearest_valid_size
from src.models.common import get_model
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg
from src.utils.periodicity import get_periodicity_dict
import pickle
from src.datamodule.seg import TestDataset, load_chunk_features, nearest_valid_size
from src.datamodule.seg_overlap import (
    TestDataset as TestDatasetOverlap,
    load_chunk_features as load_chunk_features_overlap,
)
from timm.utils import ModelEmaV2


def get_train_dataloader(cfg: DictConfig, fold: int | None, stride: int = 0) -> DataLoader:
    split = "fold_{}".format(fold) if fold is not None else "split"
    print(f"32:{fold}", split)

    series_ids = cfg[split]["train_series_ids"]

    # train data
    periodicity_dict = None
    if cfg.datamodule.zero_periodicity:
        periodicity_dict = get_periodicity_dict(cfg)

    if cfg.datamodule.how == "overlap":
        chunk_features = load_chunk_features_overlap(
            duration=cfg.duration,
            feature_names=cfg.features,
            series_ids=series_ids,
            processed_dir=Path(cfg.dir.processed_dir),
            phase="train",
            # periodicity_dict=periodicity_dict,
            stride=stride,
            overlap=cfg.datamodule.overlap,
            debug=cfg.debug,
        )
        valid_dataset = TestDatasetOverlap(cfg, chunk_features=chunk_features)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        chunk_features = load_chunk_features(
            duration=cfg.duration,
            feature_names=cfg.features,
            series_ids=series_ids,
            processed_dir=Path(cfg.dir.processed_dir),
            phase="train",
            periodicity_dict=periodicity_dict,
            stride=stride,
            debug=cfg.debug,
        )
        valid_dataset = TestDataset(cfg, chunk_features=chunk_features)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return valid_dataloader, series_ids


def inference(
    duration: int, loader: DataLoader, model: nn.Module, device: torch.device, use_amp, overlap: int | None = None
) -> tuple[list[str], np.ndarray]:
    model = model.to(device)
    model.eval()

    preds = []
    keys = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                pred = model(x)["logits"].sigmoid()
                """
                pred = resize(
                    pred.detach().cpu(),
                    size=[duration, pred.shape[2]],
                    antialias=False,
                )
                """
            key = batch["key"]
            preds.append(pred.detach().cpu().numpy())
            keys.extend(key)

    preds = np.concatenate(preds)
    l = overlap if overlap > 0 else None
    r = -overlap if overlap > 0 else None
    preds = preds[:, l:r, :]
    model.train()
    return keys, preds


class SegModel(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        val_event_df: pl.DataFrame,
        feature_dim: int,
        num_classes: int,
        duration: int,
        datamodule=None,
        fold: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df
        self.duration = duration
        valid_duration = nearest_valid_size(int(duration * cfg.upsample_rate), cfg.downsample_rate)
        self.model = get_model(
            cfg,
            feature_dim=feature_dim,
            n_classes=num_classes,
            num_timesteps=valid_duration // cfg.downsample_rate,
        )
        self.postfix = f"_fold{fold}" if fold is not None else ""
        self.validation_step_outputs: list = []
        self.__best_loss = np.inf
        self.__best_score = 0.0
        self.datamodule = datamodule
        self.epoch = 0
        self.fold = fold
        print(f"142:{fold} {self.fold}")

        self.overlap = cfg.datamodule.overlap

        self.model_ema = None
        if self.cfg.averaged_model.how == "ema":
            print("Using EMA")
            self.model_ema = ModelEmaV2(self.model, self.cfg.averaged_model.ema_decay)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x, labels)

    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, "val")

    def on_after_backward(self):
        if self.model_ema is not None:
            self.model_ema.update(self.model)

    def __share_step(self, batch, mode: str) -> torch.Tensor:
        if mode == "train":
            do_mixup = np.random.rand() < self.cfg.augmentation.mixup_prob
            do_cutmix = np.random.rand() < self.cfg.augmentation.cutmix_prob
            output = self.model(
                batch["feature"], batch["cat_feature"], batch["label"], batch["masks"], do_mixup, do_cutmix
            )
        elif mode == "val":
            do_mixup = False
            do_cutmix = False
            if self.model_ema is not None:
                output = self.model_ema.module(
                    batch["feature"], batch["cat_feature"], batch["label"], batch["masks"], do_mixup, do_cutmix
                )
            else:
                output = self.model(
                    batch["feature"], batch["cat_feature"], batch["label"], batch["masks"], do_mixup, do_cutmix
                )

        loss: torch.Tensor = output["loss"]
        logits = output["logits"]  # (batch_size, n_timesteps, n_classes)

        if mode == "train":
            self.log(
                f"{mode}_loss{self.postfix}",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        elif mode == "val":
            # アップサンプリングやダウンサンプリングで長さが変わっているのでリサイズしてもとに戻す
            if self.cfg.model.name == "CenterNet":
                resized_logits = self.model._logits_to_proba_per_step(logits, self.duration).detach().cpu()
            else:
                resized_logits = resize(
                    logits.sigmoid().detach().cpu(),
                    size=[self.duration, logits.shape[2]],
                    antialias=False,
                )
            resized_labels = resize(
                batch["label"].detach().cpu(),
                size=[self.duration, logits.shape[2]],
                antialias=False,
            )
            self.validation_step_outputs.append(
                (
                    batch["key"],
                    resized_labels.numpy(),
                    resized_logits.numpy(),
                    loss.detach().item(),
                )
            )
            self.log(
                f"{mode}_loss{self.postfix}",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        return loss

    def save_train_pred(self):
        dataloader, unique_series_ids = get_train_dataloader(self.cfg, self.fold)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keys, preds = inference(
            self.cfg.duration,
            dataloader,
            self.model,
            device,
            use_amp=self.cfg.use_amp,
            overlap=self.cfg.datamodule.overlap,
        )
        series2preds = {}
        series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
        for series_id in unique_series_ids:
            series_idx = np.where(series_ids == series_id)[0]
            this_series_preds = preds[series_idx].reshape(-1, 3)
            # 閾値以下を0にする
            this_series_preds[this_series_preds < self.cfg.label_correct.pred_threshold] = 0.0
            series2preds[series_id] = this_series_preds * self.cfg.label_correct.pred_rate
        with open(f"series2preds{self.postfix}.pickle", "wb") as f:
            pickle.dump(series2preds, f)

    def on_train_epoch_end(self):
        if self.cfg.label_correct.use:
            if self.epoch == self.cfg.label_correct.save_epoch:
                self.save_train_pred()

        self.epoch += 1
        if self.datamodule is not None:
            self.datamodule.set_now_epoch(self.epoch)
            if self.cfg.sigma_decay is not None:
                self.datamodule.set_sigma(self.datamodule.sigma * self.cfg.sigma_decay)
        if self.cfg.sleep_decay is not None:
            self.model.update_loss_fn(self.cfg.sleep_decay)

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        l = self.overlap if self.overlap > 0 else None
        r = -self.overlap if self.overlap > 0 else None
        labels = np.concatenate([x[1][:, l:r, :] for x in self.validation_step_outputs])
        preds = np.concatenate([x[2][:, l:r, :] for x in self.validation_step_outputs])
        losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = losses.mean()

        periodicity_dict = get_periodicity_dict(self.cfg)
        val_pred_df = post_process_for_seg(
            keys=keys,
            preds=preds[:, :, [1, 2]],
            score_th=self.cfg.post_process.score_th,
            distance=self.cfg.post_process.distance,
            periodicity_dict=periodicity_dict,
        )
        print(self.val_event_df.head())
        print(val_pred_df.head())
        score = event_detection_ap(self.val_event_df.to_pandas(), val_pred_df.to_pandas())
        self.log(f"val_score{self.postfix}", score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        if ((self.cfg.monitor == "val_score") and (score > self.__best_score)) or (
            (self.cfg.monitor == "val_loss") and (loss < self.__best_loss)
        ):
            np.save(f"keys{self.postfix}.npy", np.array(keys))
            np.save(f"labels{self.postfix}.npy", labels)
            np.save(f"preds{self.postfix}.npy", preds)
            val_pred_df.write_csv(f"val_pred_df{self.postfix}.csv")
            model_dict = self.model.state_dict() if self.model_ema is None else self.model_ema.module.state_dict()
            torch.save(model_dict, f"best_model{self.postfix}.pth")
            print(f"Saved best model {self.__best_score} -> {score}")
            print(f"Saved best model {self.__best_loss} -> {loss}")
            self.__best_score = score
            self.__best_loss = loss

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)

        # 1epoch分をwarmupとするための記述
        num_warmup_steps = (
            math.ceil(self.trainer.max_steps / self.cfg.epoch) * 1 if self.cfg.scheduler.use_warmup else 0
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, num_warmup_steps=num_warmup_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
