import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import pickle
from src.utils.common import pad_if_needed
from src.utils.periodicity import get_periodicity_dict


###################
# Load Functions
###################
def load_features(
    feature_names: list[str],
    cat_feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    periodicity_dict: Optional[dict[str, np.ndarray]] = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    features = {}
    cat_features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        this_cat_feature = []
        for feature_name in feature_names:
            feat = np.load(series_dir / f"{feature_name}.npy")
            # 時間系の特徴量以外はperiodicityを考慮
            if (periodicity_dict is not None) and (("_cos" not in feature_name) and ("_sin" not in feature_name)):
                feat *= 1 - periodicity_dict[series_dir.name]
            this_feature.append(feat)
        for cat_feature_name in cat_feature_names:
            feat = np.load(series_dir / f"{cat_feature_name}.npy").astype(np.int64)
            this_cat_feature.append(feat)

        features[series_dir.name] = np.stack(this_feature, axis=1)
        cat_features[series_dir.name] = np.stack(this_cat_feature, axis=1)

    return features, cat_features


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    cat_feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
    periodicity_dict: Optional[dict[str, np.ndarray]] = None,
    stride: int = 0,  # 初期値をstride だけずらしてchunkにする
    debug: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    features = {}
    cat_features = {}

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

        this_cat_feature = []
        for cat_feature_name in cat_feature_names:
            feat = np.load(series_dir / f"{cat_feature_name}.npy").astype(np.int64)
            this_cat_feature.append(feat)

        this_feature = np.stack(this_feature, axis=1)
        this_cat_feature = np.stack(this_cat_feature, axis=1)

        num_chunks = (len(this_feature) // duration) + 1
        this_feature = pad_if_needed(this_feature, stride + num_chunks * duration, pad_value=0)
        this_cat_feature = pad_if_needed(this_cat_feature, stride + num_chunks * duration, pad_value=0)
        for i in range(num_chunks):
            chunk_feature = this_feature[stride + i * duration : stride + (i + 1) * duration]
            chunk_cat_feature = this_cat_feature[stride + i * duration : stride + (i + 1) * duration]
            # chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)
            features[f"{series_id}_{i:07}"] = chunk_feature
            cat_features[f"{series_id}_{i:07}"] = chunk_cat_feature
            if debug:
                break

    return features, cat_features


###################
# Augmentation
###################
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(
    this_event_df: pd.DataFrame,
    num_frames: int,
    duration: int,
    start: int,
    end: int,
    tolerances: list[int] = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
) -> np.ndarray:
    """
    (num_frames,3) のラベルを作成
    duration は step 単位であり、num_frames は 出力につかうframe数を表しているためアップサンプリングやダウンサンプリングの影響で変化するケースがあるので注意。
    """

    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    # ざっくりと当てはまりそうな範囲でフィルタリング
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    # labelとtoleranceを考慮したマスクを作成(イベント位置からtoleranceの範囲内のみ1)
    label = np.zeros((num_frames, 3))
    masks = np.zeros((len(tolerances), num_frames, 2))
    # onset, wakeup, sleepのラベルを作成
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        if onset >= 0 and onset < num_frames:
            label[onset, 1] = 1
            for i, tolerance in enumerate(tolerances):
                masks[i, max(0, onset - tolerance) : min(num_frames, onset + tolerance), 0] = 1
        if wakeup < num_frames and wakeup >= 0:
            label[wakeup, 2] = 1
            for i, tolerance in enumerate(tolerances):
                masks[
                    i,
                    max(0, wakeup - tolerance) : min(num_frames, wakeup + tolerance),
                    1,
                ] = 1

        onset = max(0, onset)
        wakeup = min(num_frames, wakeup)
        label[onset:wakeup, 0] = 1  # sleep

    return label, masks


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]


###################
# Dataset
###################
def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        event_df: pl.DataFrame,
        features: dict[str, np.ndarray],
        cat_features: dict[str, np.ndarray],
        fold: int | None = None,
    ):
        self.cfg = cfg
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )  # columns: onset, wakeup
        self.features = features
        self.cat_features = cat_features
        self.num_features = len(cfg.features)
        self.num_cat_features = len(cfg.cat_features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

        self.sigma = self.cfg.sigma
        self.epoch = 0
        self.fold = fold
        self.train_series2preds = None

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(f"epoch: {self.epoch}")
        if self.cfg.label_correct.use and self.train_series2preds is None:
            if self.epoch > self.cfg.label_correct.save_epoch:
                suffix = f"_fold{self.fold}" if self.fold is not None else ""
                print(f"series2preds{suffix}.pickle")
                with open(f"series2preds{suffix}.pickle", "rb") as f:
                    self.train_series2preds = pickle.load(f)

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
        this_cat_feature = self.cat_features[series_id]  # (n_steps, num_features)
        n_steps = this_feature.shape[0]

        # sample background
        if (random.random() < self.cfg.bg_sampling_rate) and (series_id not in self.cfg.ignore.negative):
            pos = negative_sampling(this_event_df, n_steps)

        # crop
        start, end = random_crop(pos, self.cfg.duration, n_steps)  # pos を含む duration の範囲でランダムにcrop
        feature = this_feature[start:end]  # (duration, num_features)
        cat_feature = this_cat_feature[start:end]  # (duration, num_features)

        # upsample
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)
        cat_feature = torch.LongTensor(cat_feature.T).unsqueeze(0)  # (1, num_features, duration)
        cat_feature = resize(
            cat_feature,
            size=[self.num_cat_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        # from hard label to gaussian label
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label, masks = get_label(this_event_df, num_frames, self.cfg.duration, start, end)
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]], offset=self.cfg.offset, sigma=self.sigma
        )  # onset, wakeup のみハードラベルなのでガウシアンラベルに変換

        # 最大値が　cfg.datamodule.max_label_smoothing までになるようにminを取る
        label = np.minimum(label, self.cfg.datamodule.max_label_smoothing)

        # label correction (maxを取る)
        if self.train_series2preds is not None:
            pstart = int(start / self.cfg.duration * num_frames)
            pend = pstart + num_frames
            preds = self.train_series2preds[series_id]
            label = np.maximum(label, preds[pstart:pend])

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "cat_feature": cat_feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
            "masks": torch.FloatTensor(masks),  # (num_tolerances, pred_length, 2)
        }


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
        chunk_cat_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.chunk_cat_features = chunk_cat_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step").drop_nulls().to_pandas()
        )
        self.num_features = len(cfg.features)
        self.num_cat_features = len(cfg.cat_features)
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

        cat_feature = self.chunk_cat_features[key]
        cat_feature = torch.LongTensor(cat_feature.T).unsqueeze(0)
        cat_feature = resize(
            cat_feature,
            size=[self.num_cat_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label, masks = get_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "cat_feature": cat_feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
            "masks": torch.FloatTensor(masks),  # (num_tolerances, duration, 2)
        }


class TestDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
        chunk_cat_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.chunk_cat_features = chunk_cat_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.num_cat_features = len(cfg.cat_features)
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

        cat_feature = self.chunk_cat_features[key]
        cat_feature = torch.LongTensor(cat_feature.T).unsqueeze(0)
        cat_feature = resize(
            cat_feature,
            size=[self.num_cat_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }


###################
# DataModule
###################
class SegDataModule(LightningDataModule):
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

        self.train_features, self.train_cat_features = load_features(
            feature_names=self.cfg.features,
            cat_feature_names=sorted(list(cfg.cat_features.keys())),
            series_ids=self.train_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
            periodicity_dict=periodicity_dict,
        )

        # valid data
        self.valid_chunk_features, self.valid_chunk_cat_features = load_chunk_features(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            cat_feature_names=sorted(list(cfg.cat_features.keys())),
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
            cat_features=self.train_cat_features,
            fold=self.fold,
        )
        train_dataset.set_sigma(self.sigma)
        train_dataset.set_epoch(self.now_epoch)
        print(f"sigma: {self.sigma}, epoch: {self.now_epoch}")
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
            chunk_cat_features=self.valid_chunk_cat_features,
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
