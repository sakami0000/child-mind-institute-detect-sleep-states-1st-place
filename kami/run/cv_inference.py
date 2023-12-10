from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm
import pickle
import gc
import pickle

from src.datamodule.seg import TestDataset, load_chunk_features, nearest_valid_size
from src.datamodule.seg_overlap import (
    TestDataset as TestDatasetOverlap,
    load_chunk_features as load_chunk_features_overlap,
)
from src.models.common import get_model
from src.utils.common import trace
from src.utils.post_process import post_process_find_peaks, make_submission
from src.utils.periodicity import get_periodicity_dict
from src.utils.metrics import event_detection_ap
import ctypes
from memory_profiler import profile


def load_model(cfg: DictConfig, fold: int) -> nn.Module:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
    )

    # load weights
    if cfg.weight is not None:
        weight_path = (
            Path(cfg.dir.cv_model_dir) / cfg.weight["exp_name"] / cfg.weight["run_name"] / f"best_model_fold{fold}.pth"
        )
        model.load_state_dict(
            torch.load(weight_path),
            strict=False,  #  Unexpected key(s) in state_dict: "loss_fn.pos_weight". の回避
        )
        print('load weight from "{}"'.format(weight_path))
    return model


def get_valid_dataloader(cfg: DictConfig, fold: int, stride: int) -> DataLoader:
    series_ids = cfg[f"fold_{fold}"]["valid_series_ids"]

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
            phase=cfg.phase,
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
            phase=cfg.phase,
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
    del chunk_features
    del valid_dataset
    gc.collect()
    return valid_dataloader, series_ids


def get_test_dataloader(cfg: DictConfig, stride: int) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = Path(cfg.dir.processed_dir) / cfg.phase
    series_ids = [x.name for x in feature_dir.glob("*")]

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
            phase=cfg.phase,
            # periodicity_dict=periodicity_dict,
            stride=stride,
            overlap=cfg.datamodule.overlap,
            debug=cfg.debug,
        )
        test_dataset = TestDatasetOverlap(cfg, chunk_features=chunk_features)
        test_dataloader = DataLoader(
            test_dataset,
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
            phase=cfg.phase,
            periodicity_dict=periodicity_dict,
            stride=stride,
            debug=cfg.debug,
        )
        test_dataset = TestDataset(cfg, chunk_features=chunk_features)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    del chunk_features
    del test_dataset
    gc.collect()
    return test_dataloader, series_ids


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
                pred = resize(
                    pred.detach().cpu(),
                    size=[duration, pred.shape[2]],
                    antialias=False,
                )
            key = batch["key"]
            preds.append(pred.detach().cpu().numpy())
            keys.extend(key)

    preds = np.concatenate(preds)
    l = overlap if overlap > 0 else None
    r = -overlap if overlap > 0 else None
    preds = preds[:, l:r, :]
    return keys, preds


@hydra.main(config_path="conf", config_name="cv_inference", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    series2preds_list = []
    for fold in range(cfg.num_fold):
        with trace(f"load model fold{fold}"):
            model = load_model(cfg, fold)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TTA (ズラして推論)
        preds_tta_list = []
        for tta_id in range(cfg.num_tta):  # TTA
            if cfg.phase == "train":
                with trace("load valid dataloader"):
                    dataloader, unique_series_ids = get_valid_dataloader(
                        cfg, fold, stride=(cfg.duration // cfg.num_tta) * tta_id
                    )
            elif cfg.phase == "test":
                with trace("load test dataloader"):
                    dataloader, unique_series_ids = get_test_dataloader(
                        cfg, stride=(cfg.duration // cfg.num_tta) * tta_id
                    )

            # inference
            preds_list = []
            with trace("inference"):
                keys, preds = inference(
                    cfg.duration, dataloader, model, device, use_amp=cfg.use_amp, overlap=cfg.datamodule.overlap
                )
                preds_tta_list.append(preds)
            series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
            del dataloader
            del preds
            del keys
            torch.cuda.empty_cache()
            gc.collect()

        # シリーズごとに各TTAの予測を平均
        series2preds = {}
        for series_id in unique_series_ids:
            series_idx = np.where(series_ids == series_id)[0]
            preds_list = []
            counts_list = []
            for tta_id in range(cfg.num_tta):
                this_series_preds = preds_tta_list[tta_id][series_idx].reshape(-1, 3)
                # stride 分ずれているので元の位置に戻す(右にずらす)。左の空白は0埋めして後で割るときに無視する
                stride = (cfg.duration // cfg.num_tta) * tta_id
                this_series_preds = np.roll(this_series_preds, stride, axis=0)
                this_series_preds[:stride] = 0
                counts = np.ones(len(this_series_preds))
                counts[:stride] = 0
                preds_list.append(this_series_preds)
                counts_list.append(counts)
            series2preds[series_id] = np.sum(preds_list, axis=0) / np.sum(counts_list, axis=0)[:, None]
        series2preds_list.append(series2preds)

        del model
        del preds_tta_list
        del preds_list
        del counts_list
        del series2preds
        del series_ids
        torch.cuda.empty_cache()
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)

    series2preds = {}
    if cfg.phase == "train":
        # リストの辞書を結合
        for se2pre in series2preds_list:
            series2preds.update(se2pre)
    elif cfg.phase == "test":
        # シリーズごとに各foldの予測を平均
        for series_id in unique_series_ids:
            series2preds[series_id] = np.mean([se2pre[series_id] for se2pre in series2preds_list], axis=0)
    with open("series2preds.pkl", "wb") as f:
        pickle.dump(series2preds, f)
    del series2preds_list
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)

    with trace("load seq_df and get unique_series_ids"):
        # 結果をparquetに保存
        if cfg.phase == "train":
            seq_df = pl.read_parquet(
                Path(cfg.dir.data_dir) / "train_series.parquet", columns=["series_id", "step", "timestamp"]
            )
        elif cfg.phase == "test":
            seq_df = pl.read_parquet(
                Path(cfg.dir.data_dir) / "test_series.parquet", columns=["series_id", "step", "timestamp"]
            )
        series_count_dict = dict(seq_df.get_column("series_id").value_counts().iter_rows())
        unique_series_ids = (
            seq_df.unique("series_id", keep="first", maintain_order=True).get_column("series_id").to_list()
        )  # 順序を保ったままseries_idを取得
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)

    with trace("concat preds"):
        preds_list = []
        for series_id in unique_series_ids:
            this_series_preds = series2preds[series_id].reshape(-1, 3)
            this_series_preds = this_series_preds[: series_count_dict[series_id], :]
            preds_list.append(this_series_preds)
        del series2preds
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        preds_all = np.concatenate(preds_list, axis=0)
        del preds_list
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        pred_df = pl.DataFrame(
            [
                pl.Series(name="pred_sleep", values=preds_all[:, 0]),
                pl.Series(name="pred_onset", values=preds_all[:, 1]),
                pl.Series(name="pred_wakeup", values=preds_all[:, 2]),
            ]
        )
        pred_df.write_parquet(f"{cfg.phase}_pred.parquet")
        del preds_all
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)

    with trace("concat preds and seq_df"):
        pred_df = pl.concat([seq_df, pred_df], how="horizontal").with_columns(
            pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
        )
        del seq_df
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)

    if cfg.phase == "train":
        # スコアリング
        event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()
        periodicity_dict = get_periodicity_dict(cfg)

        with trace("make submission"):
            sub_df = make_submission(
                pred_df,
                periodicity_dict=periodicity_dict,
                height=cfg.post_process.score_th,
                distance=cfg.post_process.distance,
                pred_prefix="pred",
            )
            score = event_detection_ap(
                event_df.to_pandas(),
                sub_df.to_pandas(),
            )
            print(f"score: {score:.4f}")

    elif cfg.phase == "test":
        # make submission
        with trace("make submission"):
            periodicity_dict = get_periodicity_dict(cfg)
            sub_df = make_submission(
                pred_df,
                periodicity_dict=periodicity_dict,
                height=cfg.post_process.score_th,
                distance=cfg.post_process.distance,
                pred_prefix="pred",
            )
        sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
