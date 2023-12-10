import itertools
import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from src.utils.common import trace
from src.utils.periodicity import predict_periodicity_v2
from tqdm import tqdm

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


rolling_std_steps = [12, 60, 120, 360]
base_cols = ["enmo", "anglez"]
window_sizes = [6, 12]
n_pca = 0

FEATURE_NAMES = (
    [
        "anglez",
        "enmo",
        "anglez_diff",
        "enmo_diff",
        "anglez_series_norm",
        "enmo_series_norm",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "minute_sin",
        "minute_cos",
        "minute15_sin",
        "minute15_cos",
        "weekday_sin",
        "weekday_cos",
        "activity_count",
        "lids",
        "anglez_abs_diff_mean_24h",
        "anglez_diff_nonzero_5_mean_24h",
        "anglez_diff_nonzero_60_mean_24h",
        "lids_mean_24h",
        "anglez_diff_5min_median",
        "enmo_clip",
        "enmo_log",
    ]
    + list(
        itertools.chain.from_iterable(
            [
                [
                    f"enmo_{window_size}_rolling_mean",
                    f"enmo_log_{window_size}_rolling_mean",
                    f"anglez_{window_size}_rolling_std",
                    f"enmo_{window_size}_rolling_std",
                    f"enmo_log_{window_size}_rolling_std",
                    f"enmo_{window_size}_rolling_max",
                    f"enmo_log_{window_size}_rolling_max",
                ]
                for window_size in window_sizes
            ]
        )
    )
    + list(
        itertools.chain.from_iterable(
            [
                [
                    f"anglez_abs_diff_{minute}_median",
                    f"anglez_abs_diff_{minute}_std",
                    f"anglez_abs_diff_{minute}_mean",
                    f"anglez_abs_diff_{minute}_max",
                    f"anglez_diff_nonzero_{minute}",
                    f"anglez_diff_nonzero_{minute}_max",
                    f"anglez_diff_nonzero_{minute}_std",
                ]
                for minute in [1, 5, 60]
            ]
        )
    )
)

CATEGORICAL_FEATURE_NAMES = [
    "hour",
    "minute_15",
]


"""
    "anglez_abs_diff_var_24h",
    "anglez_diff_nonzero_5_var_24h",
    "anglez_diff_nonzero_60_var_24h",
    "lids_var_24h",
    [f"{col}_std_diff{step}" for step in rolling_std_steps for col in base_cols]
"""

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = (
        series_df.with_columns(
            # raw データはシリーズの平均と分散でnormalize
            (
                (pl.col("anglez_raw") - pl.col("anglez_raw").mean())
                / pl.col("anglez_raw").std()
            ).alias("anglez_series_norm"),
            (
                (pl.col("enmo_raw") - pl.col("enmo_raw").mean())
                / pl.col("enmo_raw").std()
            ).alias("enmo_series_norm"),
        )
        .with_columns(
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            *to_coord(pl.col("timestamp").dt.minute() % 15, 15, "minute15"),
            *to_coord(pl.col("timestamp").dt.weekday(), 7, "weekday"),
        )
        .with_columns(
            # 一つとなりとの差分
            pl.col("anglez").diff().fill_null(0).alias("anglez_diff"),
            pl.col("enmo").diff().fill_null(0).alias("enmo_diff"),
            pl.col("anglez").diff(1).fill_null(0).abs().alias("anglez_abs_diff"),
            (pl.col("anglez_raw").diff(1).fill_null(0).abs() < 0.1)
            .cast(pl.Float32)
            .alias("anglez_diff_nonzero"),
        )
        .with_columns(
            *itertools.chain.from_iterable(
                [
                    [
                        pl.col("anglez")
                        .rolling_median(minute * 60 // 5, center=True, min_periods=1)
                        .alias(f"anglez_abs_diff_{minute}_median"),
                        pl.col("anglez")
                        .rolling_std(minute * 60 // 5, center=True, min_periods=1)
                        .alias(f"anglez_abs_diff_{minute}_std"),
                        pl.col("anglez")
                        .rolling_mean(minute * 60 // 5, center=True, min_periods=1)
                        .alias(f"anglez_abs_diff_{minute}_mean"),
                        pl.col("anglez")
                        .rolling_max(minute * 60 // 5, center=True, min_periods=1)
                        .alias(f"anglez_abs_diff_{minute}_max"),
                    ]
                    for minute in [1, 5, 60]
                ]
            )
        )
        .with_columns(
            *itertools.chain.from_iterable(
                [
                    [
                        pl.col("anglez_diff_nonzero")
                        .rolling_mean(minute * 60 // 5, center=True, min_periods=1)
                        .alias(f"anglez_diff_nonzero_{minute}"),
                        pl.col("anglez_diff_nonzero")
                        .rolling_max(minute * 60 // 5, center=True, min_periods=1)
                        .alias(f"anglez_diff_nonzero_{minute}_max"),
                        pl.col("anglez_diff_nonzero")
                        .rolling_std(minute * 60 // 5, center=True, min_periods=1)
                        .alias(f"anglez_diff_nonzero_{minute}_std"),
                    ]
                    for minute in [1, 5, 60]
                ]
            )
        )
        .with_columns(
            # 10 minute moving sum over max(0, enmo - 0.02), then smoothed using moving average over a 30-min window
            pl.col("enmo")
            .map_batches(lambda x: np.maximum(x - 0.02, 0))
            .rolling_sum(10 * 60 // 5, center=True, min_periods=1)
            .rolling_mean(30 * 60 // 5, center=True, min_periods=1)
            .alias("activity_count"),
        )
        .with_columns(
            # 100/ (activity_count + 1)
            (1 / (pl.col("activity_count") + 1)).alias("lids"),
        )
        .with_columns(
            pl.col("anglez")
            .diff()
            .abs()
            .rolling_median(12 * 5, center=True)
            .fill_null(0.0)
            .alias("anglez_diff_5min_median"),
        )
        .with_columns((pl.col("enmo")).clip_max(7.0).alias("enmo_clip"))
        .with_columns((pl.col("enmo")).log1p().alias("enmo_log"))
        .with_columns(
            *itertools.chain.from_iterable(
                [
                    [
                        pl.col(["enmo", "enmo_log"])
                        .rolling_mean(window_size, center=True)
                        .fill_null(0.0)
                        .suffix(f"_{window_size}_rolling_mean"),
                        pl.col(["anglez", "enmo", "enmo_log"])
                        .rolling_std(window_size, center=True)
                        .fill_null(0.0)
                        .suffix(f"_{window_size}_rolling_std"),
                        pl.col(["enmo", "enmo_log"])
                        .rolling_max(window_size, center=True)
                        .fill_null(0.0)
                        .suffix(f"_{window_size}_rolling_max"),
                    ]
                    for window_size in window_sizes
                ]
            )
        )
    )
    """
        .with_columns(
            # 指定区間左側のstd。null は 0 で埋める
            [
                pl.col(col).rolling_std(step, center=False).fill_null(0).alias(f"{col}_std_left{step}")
                for step in rolling_std_steps
                for col in base_cols
            ]
        )
        .with_columns(
            # left をずらして right を作る
            [
                pl.col(f"{col}_std_left{step}")
                .shift_and_fill(periods=-(step - 1), fill_value=0)
                .alias(f"{col}_std_right{step}")
                for step in rolling_std_steps
                for col in base_cols
            ]
        )
        .with_columns(
            # left と right の差分
            [
                (pl.col(f"{col}_std_right{step}") - pl.col(f"{col}_std_left{step}")).alias(f"{col}_std_diff{step}")
                for step in rolling_std_steps
                for col in base_cols
            ]
        )
    """

    # 大域特徴
    one_day_steps = 24 * 60 * 60 // 5
    series_df = series_df.with_columns(
        pl.arange(0, series_df.height).mod(one_day_steps).alias("group_number"),
    )
    # groupごとに平均を取る
    series_df = series_df.with_columns(
        pl.col("anglez_abs_diff")
        .mean()
        .over("group_number")
        .alias("anglez_abs_diff_mean_24h"),
        pl.col("anglez_diff_nonzero_5")
        .mean()
        .over("group_number")
        .alias("anglez_diff_nonzero_5_mean_24h"),
        pl.col("anglez_diff_nonzero_60")
        .mean()
        .over("group_number")
        .alias("anglez_diff_nonzero_60_mean_24h"),
        pl.col("lids").mean().over("group_number").alias("lids_mean_24h"),
    )
    """
    series_df = series_df.with_columns(
        pl.col("anglez_abs_diff").var().over("group_number").fill_null(0).alias("anglez_abs_diff_var_24h"),
        pl.col("anglez_diff_nonzero_5").var().over("group_number").fill_null(0).alias("anglez_diff_nonzero_5_var_24h"),
        pl.col("anglez_diff_nonzero_60")
        .var()
        .over("group_number")
        .fill_null(0)
        .alias("anglez_diff_nonzero_60_var_24h"),
        pl.col("lids").var().over("group_number").fill_null(0).alias("lids_var_24h"),
    )
    """

    # categorical
    series_df = series_df.with_columns(
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.minute() % 15).alias("minute_15"),
    )

    return series_df


def save_each_series(
    cfg, this_series_df: pl.DataFrame, columns: list[str], output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)

    # periodicity
    seq = this_series_df.get_column("enmo_raw").to_numpy(zero_copy_only=True)
    periodicity = predict_periodicity_v2(
        seq,
        cfg.periodicity.downsample_rate,
        cfg.periodicity.stride_min,
        cfg.periodicity.split_min,
    )
    np.save(output_dir / "periodicity.npy", periodicity)
    np.save(output_dir / "non_periodicity.npy", 1 - periodicity)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                # 全体の平均・標準偏差から標準化
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
                # raw
                pl.col("anglez").alias("anglez_raw"),
                pl.col("enmo").alias("enmo_raw"),
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("timestamp"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("anglez_raw"),
                    pl.col("enmo_raw"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(
            series_df.group_by("series_id"), total=n_unique
        ):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(
                cfg,
                this_series_df,
                FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES,
                series_dir,
            )


if __name__ == "__main__":
    main()
