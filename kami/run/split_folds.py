import os
from pathlib import Path

import hydra
import polars as pl
import yaml
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold


def make_fold(cfg: DictConfig, n_folds: int = 5):
    series_ids = [
        str(path).split("/")[-1]
        for path in (Path(cfg.dir.processed_dir) / "train").glob("*")
    ]

    event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv").drop_nulls()
    q_cut_num = 10  # no_event は含まないので全体では11クラス
    event_count_df = event_df.group_by("series_id").count()
    event_count_df = event_count_df.select(
        pl.col("series_id"),
        pl.col("count"),
        pl.col("count")
        .qcut(q_cut_num, labels=[str(i) for i in range(1, q_cut_num + 1)])
        .alias("class")
        .cast(pl.Utf8),
    )

    no_event = [
        si
        for si in series_ids
        if si not in list(event_count_df.get_column("series_id"))
    ]
    no_event_df = pl.DataFrame(
        {
            "series_id": no_event,
            "count": [0 for _ in range(len(no_event))],
            "class": [str(0) for _ in range(len(no_event))],
        }
    )
    no_event_df = no_event_df.select(
        pl.col("series_id"),
        pl.col("count").cast(pl.UInt32),
        pl.col("class").cast(pl.Utf8),
    )

    # 二つを結合
    all_df = pl.concat([event_count_df, no_event_df]).sort(by="series_id")

    X = all_df.drop("class")
    y = all_df.get_column("class")

    skf = StratifiedKFold(n_splits=n_folds)

    os.makedirs(Path(cfg.dir.input_dir) / "folds", exist_ok=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        fold_dict = {
            "train_series_ids": list(all_df.get_column("series_id").take(train_index)),
            "valid_series_ids": list(all_df.get_column("series_id").take(test_index)),
        }

        with open(Path(cfg.dir.input_dir) / f"folds/stratify_fold_{i}.yaml", "w") as wf:
            yaml.dump(fold_dict, wf)

        with open(
            Path(cfg.dir.kami_dir) / f"run/conf/split/stratify_fold_{i}.yaml", "w"
        ) as wf:
            yaml.dump(fold_dict, wf)


@hydra.main(config_path="conf", config_name="split_folds", version_base="1.2")
def main(cfg: DictConfig):
    make_fold(cfg)


if __name__ == "__main__":
    main()
