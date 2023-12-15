import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig

from src.utils.periodicity import get_periodicity_dict

from src.utils.common import (
    gaussian_label,
    nearest_valid_size,
    negative_sampling,
    pad_if_needed,
    random_crop,
)


###################
# Load Functions
###################
def load_features(
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    periodicity_dict: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            feat = np.load(series_dir / f"{feature_name}.npy")
            # 時間系の特徴量以外はperiodicityを考慮
            if (periodicity_dict is not None) and (("_cos" not in feature_name) and ("_sin" not in feature_name)):
                feat *= 1 - periodicity_dict[series_dir.name]
            this_feature.append(feat)

        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    periodicity_dict: Optional[dict[str, np.ndarray]] = None,
    stride: int = 0,  # 初期値をstride だけずらしてchunkにする
    debug: bool = False,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            feat = np.load(series_dir / f"{feature_name}.npy")
            # 時間系の特徴量以外はperiodicityを考慮
            if (periodicity_dict is not None) and (("_cos" not in feature_name) and ("_sin" not in feature_name)):
                feat *= 1 - periodicity_dict[series_dir.name]
            this_feature.append(feat)

        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        this_feature = pad_if_needed(this_feature, stride + num_chunks * duration, pad_value=0)
        for i in range(num_chunks):
            chunk_feature = this_feature[stride + i * duration : stride + (i + 1) * duration]
            # chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)
            features[f"{series_id}_{i:07}"] = chunk_feature
            if debug:
                break

    return features  # type: ignore


###################
# Label
###################
def get_centernet_label(
    this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int
) -> np.ndarray:
    # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    # labelを作成
    # onset_pos, wakeup_pos, onset_offset, wakeup_offset, onset_bbox_size, wakeup_bbox_size
    label = np.zeros((num_frames, 6))
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset_pos = int((onset - start) / duration * num_frames)
        onset_offset = (onset - start) / duration * num_frames - onset_pos
        wakeup_pos = int((wakeup - start) / duration * num_frames)
        wakeup_offset = (wakeup - start) / duration * num_frames - wakeup_pos

        # 区間に入らない場合は、posをclipする.
        # e.g. num_frames=100, onset_pos=50, wakeup_pos=150
        # -> onset_pos=50, wakeup_pos=100, bbox_size=(100-50)/100=0.5
        bbox_size = (min(wakeup_pos, num_frames) - max(onset_pos, 0)) / num_frames

        if onset_pos >= 0 and onset_pos < num_frames:
            label[onset_pos, 0] = 1
            label[onset_pos, 2] = onset_offset
            label[onset_pos, 4] = bbox_size

        if wakeup_pos < num_frames and wakeup_pos >= 0:
            label[wakeup_pos, 1] = 1
            label[wakeup_pos, 3] = wakeup_offset
            label[wakeup_pos, 5] = bbox_size

    # org_pos = pred_pos + pred_offset, don't use bbox_size. it's for loss.
    return label


class TrainDataset(Dataset):
    def __init__(
        self,
        cfg,
        features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )
        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

        self.sigma = self.cfg.sigma

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        pos = self.event_df.at[idx, event]
        series_id = self.event_df.at[idx, "series_id"]
        self.event_df["series_id"]
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
        # extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        n_steps = this_feature.shape[0]

        # sample background
        if random.random() < self.cfg.bg_sampling_rate:
            pos = negative_sampling(this_event_df, n_steps)

        # crop
        if n_steps > self.cfg.duration:
            start, end = random_crop(pos, self.cfg.duration, n_steps)
            feature = this_feature[start:end]
        else:
            start, end = 0, self.cfg.duration
            feature = pad_if_needed(this_feature, self.cfg.duration)

        # upsample
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        # from hard label to gaussian label
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_centernet_label(this_event_df, num_frames, self.cfg.duration, start, end)
        label[:, [0, 1]] = gaussian_label(label[:, [0, 1]], offset=self.cfg.offset, sigma=self.cfg.sigma)

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
            "masks": torch.FloatTensor(0),
        }


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_centernet_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
            "masks": torch.FloatTensor(0),
        }


class CenterNetTestDataset(Dataset):
    def __init__(
        self,
        cfg,
        chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }


###################
# DataModule
###################
class CenterNetDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig, fold: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv").drop_nulls()
        self.fold = fold

        if self.fold is None:  # single fold
            self.train_series_ids = self.cfg.split.train_series_ids
            self.valid_series_ids = self.cfg.split.valid_series_ids
        else:
            self.train_series_ids = self.cfg[f"fold_{fold}"].train_series_ids
            self.valid_series_ids = self.cfg[f"fold_{fold}"].valid_series_ids

        self.train_event_df = self.event_df.filter(pl.col("series_id").is_in(self.train_series_ids)).filter(
            ~pl.col("series_id").is_in(self.cfg.ignore.train)
        )
        self.valid_event_df = self.event_df.filter(pl.col("series_id").is_in(self.valid_series_ids))

        # train data
        periodicity_dict = None
        if self.cfg.datamodule.zero_periodicity:
            periodicity_dict = get_periodicity_dict(self.cfg)

        self.train_features = load_features(
            feature_names=self.cfg.features,
            series_ids=self.train_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
            periodicity_dict=periodicity_dict,
        )

        # valid data
        self.valid_chunk_features = load_chunk_features(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            series_ids=self.valid_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
            periodicity_dict=periodicity_dict,
        )
        self.sigma = cfg.sigma

        self.now_epoch = 0

    def set_now_epoch(self, epoch):
        self.now_epoch = epoch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
        )
        train_dataset.set_sigma(self.sigma)
        print(f"sigma: {self.sigma}")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            chunk_features=self.valid_chunk_features,
            event_df=self.valid_event_df,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
