import glob
import logging
import os
import pickle
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from components.factories.tree_factory import CatModel
from components.fast_metrics import event_detection_ap
from components.utils import post_process_from_2nd
from google.cloud import storage
from omegaconf import DictConfig
from scipy.signal import find_peaks
from tqdm import tqdm

plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_submission(
    row_idxs: dict[str, str],
    preds: np.ndarray,
    steps: np.ndarray,
    timestamps: np.ndarray,
    height: float = 0.001,
    distance: int = 100,
) -> pl.DataFrame:
    records = []
    for row in tqdm(row_idxs):
        _preds = preds[row["row_id"]]
        _steps = steps[row["row_id"]]
        _timestamps = timestamps[row["row_id"]]
        for i, event_name in enumerate(["onset", "wakeup"]):
            idxs, peak_heights = find_peaks(_preds[:, i], height=height, distance=distance)
            peak_steps = _steps[idxs]
            peak_timestamps = _timestamps[idxs]
            if len(peak_steps) > 0:
                records.append(
                    {
                        "series_id": row["series_id"],
                        "event": event_name,
                        "step": peak_steps,
                        "timestamp": peak_timestamps,
                        "score": peak_heights["peak_heights"],
                    }
                )
    # sub_df = pl.DataFrame(records)
    sub_df = pl.from_pandas(pd.DataFrame(records).explode(["step", "timestamp", "score"]))
    return sub_df


def prepair_dir(config: DictConfig) -> None:
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def upload_directory(store_config: DictConfig) -> None:
    storage_client = storage.Client(store_config.gcs_project)
    bucket = storage_client.get_bucket(store_config.bucket_name)
    glob._ishidden = lambda x: False
    filenames = glob.glob(os.path.join(store_config.save_path, "**"), recursive=True)
    for filename in filenames:
        if os.path.isdir(filename):
            continue
        destination_blob_name = os.path.join(
            store_config.gcs_path,
            filename.split(store_config.save_path)[-1][1:],
        )
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(filename)


@hydra.main(config_path="../../yamls", config_name="stacking.yaml")
def main(config: DictConfig):
    prepair_dir(config)
    set_seed(config.seed)
    with open(f"./input/stacking/{config.feature.version}.pkl", "rb") as f:
        df: pl.DataFrame = pickle.load(f)
        # 元データに存在しない未来の部分を削除
        df = df.filter(pl.col("timestamp").is_not_null())
        df = df.with_columns(pl.col(config.feature.cat_cols).cast(int))

    config.catboost.categorical_features_indices = [
        idx for idx, col in enumerate(config.feature.numerical_cols + config.feature.cat_cols) if col in config.feature.cat_cols
    ]

    for event in ["onset", "wakeup"]:
        df = df.with_columns((pl.col("event") == event).cast(int).fill_null(0).alias("label"))
        print(df["label"].value_counts())
        model = CatModel(config.catboost)
        df = model.cv(df)
        model.save_importance(config.store.result_path, suffix=f"_{event}")
        model.save_model(config.store.model_path, suffix=f"_{event}")
        df.select(["series_id", "chunk_id", "step", config.feature.label_col, config.feature.pred_col]).write_parquet(
            f"{config.store.result_path}/pred_{event}.parquet"
        )

    pred_df = (
        pl.read_parquet(f"{config.store.result_path}/pred_onset.parquet")
        .rename({"label_pred": "stacking_prediction_onset"})
        .drop("label")
        .join(
            pl.read_parquet(f"{config.store.result_path}/pred_wakeup.parquet").rename({"label_pred": "stacking_prediction_wakeup"}).drop("label"),
            on=["series_id", "step"],
            how="left",
        )
    )
    pred_df = pred_df.join(df[["series_id", "step", "timestamp", "event"]], on=["series_id", "step"])
    pred_df = pred_df.with_columns(pl.Series("row_id", range(len(pred_df))))
    row_idxs = pred_df.group_by(["series_id", "chunk_id"]).agg(pl.col(["row_id"])).rows(named=True)
    preds = pred_df[["stacking_prediction_onset", "stacking_prediction_wakeup"]].to_numpy()
    steps = pred_df["step"].to_numpy()
    timestamps = pred_df["timestamp"].to_numpy()

    sub_df = make_submission(row_idxs, preds, steps, timestamps, distance=8, height=0.001)
    mean_ap = event_detection_ap(
        pl.read_csv("./input/train_events.csv").filter(pl.col("step").is_not_null()).to_pandas(),
        sub_df.to_pandas(),
    )
    logger.info(f"1minute score: {mean_ap}")
    sub_df = post_process_from_2nd(pred_df)
    mean_ap = event_detection_ap(
        pl.read_csv("./input/train_events.csv").filter(pl.col("step").is_not_null()).to_pandas(),
        sub_df.to_pandas(),
    )
    logger.info(f"postprocess score: {mean_ap}")
    if config.store.gcs_project is not None:
        upload_directory(config.store)


if __name__ == "__main__":
    main()
