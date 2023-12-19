import gc
import os
import pickle
import re
import time
from contextlib import contextmanager

import polars as pl
import yaml
from components.metrics import score
from components.preprocess.feature_creator import (
    LIDFeatureCreator,
    TimeFeatureCreator,
    TimeSeriesDiffFeatureCreator,
    TimeSeriesStatsDiffFeatureCreator,
    TimeSeriesStatsFeatureCreator,
)
from components.stacking.io import load_kami_data, load_sakami_data
from components.utils import load_periodicity, make_submission


@contextmanager
def timer(name):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def load_data() -> pl.DataFrame:
    series_df = pl.scan_parquet("./input/train_series.parquet")
    series_df = series_df.with_columns(pl.col("timestamp").str.to_datetime())
    series_df = series_df.with_columns(
        pl.col("timestamp").alias("raw_timestamp"), pl.col("timestamp").dt.truncate("1m")
    )
    event_df = pl.scan_csv("./input/train_events.csv")
    event_df = event_df.with_columns(pl.col("timestamp").str.to_datetime())

    series_df = series_df.join(
        event_df.select(["series_id", "timestamp", "event"]),
        on=["series_id", "timestamp"],
        how="left",
    )

    return series_df.collect()


def make_chunk_data(sub_df: pl.DataFrame, step_size: int = 360) -> pl.DataFrame:
    chunk_df = sub_df[["series_id", "step"]]
    chunk_df = chunk_df.with_columns(
        pl.col("step").map_elements(lambda x: range(x - step_size, x + step_size, 12))
    )
    chunk_df = chunk_df.explode("step").unique()
    chunk_df = chunk_df.sort(["series_id", "step"])
    chunk_df = chunk_df.filter(pl.col("step") >= 0)
    chunk_df = chunk_df.with_columns(pl.col("step").cast(pl.Int64))
    chunk_df = chunk_df.with_columns(
        ((pl.col("step") - pl.col("step").shift(1)) != 12)
        .cast(int)
        .cumsum()
        .over("series_id")
        .fill_null(0)
        .alias("chunk_id")
    )

    return chunk_df


def main() -> None:
    """
    1 minuteでtruncateする
    """
    VERSION = re.split("[.]", os.path.basename(__file__))[-2]
    os.makedirs("./input/stacking", exist_ok=True)

    with timer("load_data"):
        series_df = load_data()
        sakami_model_names = [
            "148_gru_scale_factor",
            "156_gru_transformer_residual",
            "163_gru_sleep_target",
            "179_gru_minute_embedding_sync",
        ]
        kami_model_names = ["exp068_transformer", "exp078_lstm", "exp081_mixup_short_feat14"]
        # shimacos_model_names = ["exp076_072_only_periodicity", "exp077_073_deberta_v3_small_gru"]
        # shimacos_df = load_shimacos_data(shimacos_model_names)
        # shimacos_df = shimacos_df.with_columns(pl.col("step").cast(pl.UInt32))
        # series_df = series_df.join(shimacos_df, on=["series_id", "step"], how="left")
        sakami_df = load_sakami_data(sakami_model_names)
        kami_df = load_kami_data(kami_model_names)
        model_names = sakami_model_names + kami_model_names
        _pred_df = pl.concat([sakami_df, kami_df, series_df], how="horizontal")
        pred_df = _pred_df.group_by(["series_id", "timestamp"], maintain_order=True).agg(
            pl.col([f"pred_onset_{model_name}" for model_name in model_names]).mean(),
            pl.col([f"pred_wakeup_{model_name}" for model_name in model_names]).mean(),
            pl.col(
                [
                    f"pred_sleep_{model_name}"
                    for model_name in kami_model_names + ["163_gru_sleep_target"]
                ]
            ).mean(),
            pl.col(["anglez", "enmo"]).mean(),
            pl.col("event").first(),
        )
        _pred_df = (
            _pred_df.drop("timestamp")
            .rename({"raw_timestamp": "timestamp"})
            .select(
                pl.col(["series_id", "step", "timestamp"]),
                pl.col(
                    [f"pred_onset_{model_name}" for model_name in model_names]
                    + [f"pred_wakeup_{model_name}" for model_name in model_names]
                ).suffix("_raw"),
            )
        )
        pred_df = pred_df.join(_pred_df, on=["series_id", "timestamp"], how="left")
        pred_df = pred_df.with_columns(
            (
                pl.sum_horizontal([f"pred_onset_{model_name}" for model_name in model_names])
                / len(model_names)
            ).alias("prediction_onset"),
            (pl.min_horizontal([f"pred_onset_{model_name}" for model_name in model_names])).alias(
                "prediction_onset_min"
            ),
            (pl.max_horizontal([f"pred_onset_{model_name}" for model_name in model_names])).alias(
                "prediction_onset_max"
            ),
            (
                pl.sum_horizontal([f"pred_wakeup_{model_name}" for model_name in model_names])
                / len(model_names)
            ).alias("prediction_wakeup"),
            (pl.min_horizontal([f"pred_wakeup_{model_name}" for model_name in model_names])).alias(
                "prediction_wakeup_min"
            ),
            (pl.max_horizontal([f"pred_wakeup_{model_name}" for model_name in model_names])).alias(
                "prediction_wakeup_max"
            ),
        ).with_columns(
            (
                pl.sum_horizontal(
                    [
                        (pl.col("prediction_onset") - pl.col(f"pred_onset_{model_name}")) ** 2
                        for model_name in model_names
                    ]
                )
                / len(model_names)
            ).alias("prediction_onset_var"),
            (
                pl.sum_horizontal(
                    [
                        (pl.col("prediction_wakeup") - pl.col(f"pred_wakeup_{model_name}")) ** 2
                        for model_name in model_names
                    ]
                )
                / len(model_names)
            ).alias("prediction_wakeup_var"),
        )

        periodicity_df = load_periodicity().collect()
        pred_df = pred_df.with_columns(pl.col("step").cast(pl.Int64))
        pred_df = pred_df.join(periodicity_df, on=["series_id", "step"], how="left")
        del series_df, sakami_df, kami_df, periodicity_df
        gc.collect()
        fold_df = pl.read_parquet("./input/preprocess/fold_v2.parquet")
        # 周期性削除
        _df = pred_df.filter(pl.col("periodicity") == 0)
        _df = _df.with_columns(pl.Series("row_id", range(len(_df))))

        row_idxs = _df.group_by(["series_id"]).agg(pl.col(["row_id"])).rows(named=True)
        preds = _df[["prediction_onset", "prediction_wakeup"]].to_numpy()
        steps = _df["step"].to_numpy()
        timestamps = _df["timestamp"].to_numpy()

        sub_df = make_submission(row_idxs, preds, steps, timestamps, distance=8)
        mean_ap, matched_df = score(
            pl.read_csv("./input/train_events.csv"), sub_df, return_detections_matched=True
        )
        print(mean_ap)
        matched_df.to_csv(f"./input/stacking/{VERSION}_matched.csv", index=False)
        sub_df = sub_df.filter(pl.col("step") != 0)

    with timer("create chunk data"):
        chunk_df = make_chunk_data(sub_df, step_size=96)
        chunk_df = chunk_df.join(pred_df, on=["series_id", "step"], how="left")
        chunk_df = chunk_df.with_columns(
            (pl.col("event") == "onset").cast(pl.Int64).fill_null(0).alias("label_onset"),
            (pl.col("event") == "wakeup").cast(pl.Int64).fill_null(0).alias("label_wakeup"),
        )
        print(
            "recall:",
            len(chunk_df.filter(pl.col("event").is_not_null()))
            / len(pred_df.filter(pl.col("event").is_not_null())),
        )

        # ===============
        #   特徴量生成
        # ===============
        with timer("create lid"):
            creator = LIDFeatureCreator(
                [], timeseries_cols=["series_id", "chunk_id"], window_size=5
            )
            chunk_df = creator.create(chunk_df)

        with timer("create timeseries features"):
            for creator in [
                TimeSeriesStatsFeatureCreator,
                TimeSeriesStatsDiffFeatureCreator,
                TimeSeriesDiffFeatureCreator,
                TimeFeatureCreator,
            ]:
                chunk_df = creator(
                    signal_names=[
                        "enmo",
                        "anglez",
                        "lid",
                    ],
                    window_sizes=[3],
                    timeseries_cols=["series_id", "chunk_id"],
                ).create(chunk_df)
        with timer("create timeseries features (pred)"):
            for creator in [
                TimeSeriesStatsDiffFeatureCreator,
                TimeSeriesDiffFeatureCreator,
            ]:
                chunk_df = creator(
                    signal_names=[f"pred_onset_{model_name}" for model_name in model_names]
                    + [f"pred_wakeup_{model_name}" for model_name in model_names],
                    window_sizes=[3, 5],
                    timeseries_cols=["series_id", "chunk_id"],
                ).create(chunk_df)
        cat_cols = ["time_idx", "hour", "minute", "minute_mod15", "weekday"]
        feature_cols = [
            col
            for col in chunk_df.columns
            if col
            not in [
                "series_id",
                "step",
                "timestamp",
                "night",
                "event",
                "label",
                "fold",
                "chunk_id",
                "peak_step",
                "label_onset",
                "label_wakeup",
            ]
            + cat_cols
        ]
        feature_dict = {
            "version": VERSION,
            "label_col": "label",
            "label_cols": ["label_onset", "label_wakeup"],
            "pred_col": "label_pred",
            "pred_cols": ["label_onset_pred", "label_wakeup_pred"],
            "numerical_cols": feature_cols,
            "cat_cols": cat_cols,
        }
        with open(f"./yamls/feature/stacking_{VERSION}.yaml", "w") as f:
            yaml.dump(feature_dict, f)
        print("feature_cols:", feature_cols)
        with timer("save"):
            chunk_df = chunk_df.join(fold_df, on=["series_id"], how="left")
            with open(f"./input/stacking/{VERSION}.pkl", "wb") as f:
                pickle.dump(chunk_df, f)
            chunk_df.write_parquet(f"./input/stacking/{VERSION}.parquet")


if __name__ == "__main__":
    main()
