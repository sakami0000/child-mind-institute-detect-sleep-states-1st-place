import os

import numpy as np
import polars as pl
import yaml


def load_data() -> pl.DataFrame:
    event_df = pl.read_csv("./input/train_events.csv")
    event_df = event_df.with_columns(pl.col("timestamp").str.to_datetime())
    event_df = event_df.with_columns(pl.col("step").cast(pl.UInt32))
    return event_df


def main():
    os.makedirs("./input/preprocess", exist_ok=True)
    event_df = load_data()
    event_df = event_df[["series_id"]].unique()
    event_df = event_df.with_columns(pl.Series("row_id", np.arange(len(event_df))))

    folds = np.zeros(len(event_df))
    for i in range(5):
        with open(f"./input/folds/stratify_fold_{i}.yaml", "r") as f:
            stratify_fold = yaml.safe_load(f)
            valid_ids = stratify_fold["valid_series_ids"]
        idx = event_df.filter(pl.col("series_id").is_in(valid_ids))["row_id"].to_numpy()
        folds[idx] = i

    event_df = event_df.with_columns(pl.Series("fold", folds))
    event_df[["series_id", "fold"]].write_parquet("./input/preprocess/fold_v2.parquet")


if __name__ == "__main__":
    main()
