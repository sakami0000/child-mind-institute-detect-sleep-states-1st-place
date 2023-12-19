import logging
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(name)s] [L%(lineno)d] [%(levelname)s][%(funcName)s] %(message)s "
    )
)
logger.addHandler(handler)


class SleepDataset(object):
    def __init__(self, config: DictConfig, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load
        train_df = pl.read_parquet(config.train_path)

        self.id_col = config.id_col
        self.label_col = config.label_col
        # preprocessing

        if mode == "train":
            df = train_df.filter(pl.col("fold") != config.n_fold)
            self.labels = df[self.label_col].to_list()
        elif mode == "valid":
            df = train_df.filter(pl.col("fold") == config.n_fold)
            self.labels = df[self.label_col].to_list()
        self.ids = df[self.id_col].to_list()
        self.features_dict = {col: df[col].to_list() for col in config.feature.numerical_cols}
        self.steps = df["step"].to_list()
        self.numerial_cols = config.feature.numerical_cols
        self.window_size = config.window_size
        self.chunk_size = config.chunk_size
        self.normalize = config.normalize
        self.ids, self.features_dict, self.steps, self.labels = self._preprocess(
            self.ids, self.features_dict, self.steps, self.labels
        )
        self.seq_lens = [
            len(feature) for feature in self.features_dict[config.feature.numerical_cols[0]]
        ]

    def _preprocess(self, ids, features_dict, steps_list, labels_list):
        _features_dict = {col: [] for col in self.numerial_cols}
        _ids = []
        _steps = []
        _labels = []
        for batch_idx, (id_, steps, labels) in enumerate(
            zip(ids, steps_list, labels_list, strict=True)
        ):
            if len(labels) > self.chunk_size:
                for i in range(0, len(labels), self.window_size):
                    _ids.append(id_)
                    seq_len = len(labels[i : i + self.chunk_size])
                    if seq_len < self.chunk_size:
                        _steps.append(
                            steps[i : i + self.chunk_size] + [-1] * (self.chunk_size - seq_len)
                        )
                        _labels.append(
                            labels[i : i + self.chunk_size] + [-1] * (self.chunk_size - seq_len)
                        )
                        for col in self.numerial_cols:
                            _features_dict[col].append(
                                features_dict[col][batch_idx][i : i + self.chunk_size]
                                + [-1] * (self.chunk_size - seq_len)
                            )
                    else:
                        _steps.append(steps[i : i + self.chunk_size])
                        _labels.append(labels[i : i + self.chunk_size])
                        for col in self.numerial_cols:
                            _features_dict[col].append(
                                features_dict[col][batch_idx][i : i + self.chunk_size]
                            )
            else:
                _ids.append(id_)
                _steps.append(steps)
                _labels.append(labels)
                for col in self.numerial_cols:
                    _features_dict[col].append(features_dict[col][batch_idx])

        return _ids, _features_dict, _steps, _labels

    def set_refinement_step(self):
        self.refinement_step = True

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = {self.id_col: self.ids[idx]}
        self.seq_lens[idx]
        if self.mode == "train":
            random_idx = 0
        else:
            random_idx = 0
        features = np.stack(
            [
                self.features_dict[col][idx][random_idx : random_idx + self.chunk_size]
                for col in self.numerial_cols
            ]
        )
        out.update(
            {
                "numerical_feature": features,
                "step": np.array(self.steps[idx][random_idx : random_idx + self.chunk_size]),
            }
        )
        if self.normalize:
            std = StandardScaler()
            out["numerical_feature"] = std.fit_transform(out["numerical_feature"])
        if self.mode != "test":
            out.update(
                {
                    self.label_col: np.array(
                        self.labels[idx][random_idx : random_idx + self.chunk_size]
                    )
                }
            )
            return out
        return out


class SleepDatasetV2(object):
    def __init__(self, config: DictConfig, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load
        train_df = pl.scan_parquet(config.train_path, low_memory=True, row_count_name="row_idx")
        features = np.load(config.features_path)

        self.id_col = config.id_col
        self.label_col = config.label_col
        # preprocessing

        if mode == "train":
            df = train_df.filter(pl.col("fold") != config.n_fold)
            self.labels = df.select(self.label_col).collect().to_numpy().flatten()
            self.numerical_features = features[df.select("row_idx").collect().to_numpy().flatten()]
        elif mode == "valid":
            df = train_df.filter(pl.col("fold") == config.n_fold)
            self.labels = df.select(self.label_col).collect().to_numpy().flatten()
            self.numerical_features = features[df.select("row_idx").collect().to_numpy().flatten()]
        id_df = df.select(self.id_col).collect()
        row_idxs_list = (
            id_df.with_columns(pl.Series("row_idx", np.arange(len(id_df))))
            .group_by(self.id_col, maintain_order=True)
            .agg(pl.col("row_idx"))["row_idx"]
            .to_numpy()
        )
        self.ids = id_df.to_numpy().flatten()
        if mode == "train":
            std = StandardScaler()
            self.numerical_features = std.fit_transform(self.numerical_features)
            with open(config.std_path, "wb") as f:
                pickle.dump(std, f)
        else:
            with open(config.std_path, "rb") as f:
                std = pickle.load(f)
            self.numerical_features = std.transform(self.numerical_features)

        self.steps = df.select("step").collect().to_numpy().flatten()
        self.numerial_cols = config.feature.numerical_cols
        self.window_size = config.window_size
        self.chunk_size = config.chunk_size
        self.normalize = config.normalize

        self.indices = self._create_indices(row_idxs_list)

    def _create_indices(self, row_idxs_list):
        indices = []
        for row_idxs in row_idxs_list:
            if len(row_idxs) > self.chunk_size:
                for i in range(0, len(row_idxs), self.window_size):
                    indices.append(row_idxs[i : i + self.chunk_size])
            else:
                indices.append(row_idxs)
        return indices

    def set_refinement_step(self):
        self.refinement_step = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        indices = self.indices[idx]
        ids = self.ids[indices].tolist()
        steps = self.steps[indices]
        numerical_features = self.numerical_features[indices]
        seq_len = len(numerical_features)
        if seq_len < self.chunk_size:
            numerical_features = np.pad(
                numerical_features, ((0, self.chunk_size - seq_len), (0, 0))
            )
            steps = np.pad(steps, (0, self.chunk_size - seq_len), constant_values=-1)
        out = {
            self.id_col: ids[0],
            "numerical_feature": numerical_features.astype(np.float32),
            "step": steps.astype(np.float32),
        }
        # if self.normalize:
        #     std = StandardScaler()
        #     out["numerical_feature"] = std.fit_transform(out["numerical_feature"])
        if self.mode != "test":
            labels = self.labels[indices]
            if seq_len < self.chunk_size:
                labels = np.pad(labels, (0, self.chunk_size - seq_len), constant_values=-1)
            out.update({self.label_col: labels})
        return out


class MultiLabelSleepDataset(object):
    def __init__(self, config: DictConfig, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load
        logger.info("start load data")
        train_df = pl.read_parquet(config.train_path, low_memory=True, row_count_name="row_idx")
        features = np.load(config.features_path)

        self.id_col = config.id_col
        self.label_col = config.label_col
        self.label_cols = config.feature.label_cols
        # preprocessing
        logger.info("start collect")
        if mode == "train":
            df = train_df.filter(pl.col("fold") != config.n_fold)
            self.labels = df.select(self.label_cols).to_numpy()
            self.numerical_features = features[
                df.select("row_idx").to_numpy().flatten(),
                : len(config.feature.numerical_cols),
            ]
            self.cat_features = features[
                df.select("row_idx").to_numpy().flatten(),
                len(config.feature.numerical_cols) :,
            ]
        elif mode == "valid":
            df = train_df.filter(pl.col("fold") == config.n_fold)
            self.labels = df.select(self.label_cols).to_numpy()
            self.numerical_features = features[
                df.select("row_idx").to_numpy().flatten(),
                : len(config.feature.numerical_cols),
            ]
            self.cat_features = features[
                df.select("row_idx").to_numpy().flatten(),
                len(config.feature.numerical_cols) :,
            ]
        id_df = df.select(self.id_col)
        row_idxs_list = (
            id_df.with_columns(pl.Series("row_idx", np.arange(len(id_df))))
            .group_by(self.id_col, maintain_order=True)
            .agg(pl.col("row_idx"))["row_idx"]
            .to_numpy()
        )
        self.ids = id_df.to_numpy().flatten()
        logger.info("start normalize")
        if config.normalize:
            if mode == "train":
                std = StandardScaler()
                self.numerical_features = std.fit_transform(self.numerical_features)
                with open(config.std_path, "wb") as f:
                    pickle.dump(std, f)
            else:
                with open(config.std_path, "rb") as f:
                    std = pickle.load(f)
                self.numerical_features = std.transform(self.numerical_features)
        logger.info("finish normalize")

        self.steps = df.select("step").to_numpy().flatten()
        self.numerial_cols = config.feature.numerical_cols
        self.window_size = config.window_size
        self.chunk_size = config.chunk_size
        self.normalize = config.normalize

        logger.info("start create indices")
        self.indices = self._create_indices(
            row_idxs_list, self.config.use_only_positive_chunk, self.config.use_only_non_periodicity
        )
        logger.info("finish create indices")

    def _create_indices(
        self,
        row_idxs_list,
        use_only_positive_chunk: bool = False,
        use_only_non_periodicity: bool = False,
    ):
        indices = []
        periodicity_idx = self.config.feature.cat_cols.index("periodicity")
        for row_idxs in row_idxs_list:
            if len(row_idxs) > self.chunk_size:
                for i in range(0, len(row_idxs), self.window_size):
                    idxs = row_idxs[i : i + self.chunk_size]
                    if len(idxs) < self.chunk_size:
                        idxs = row_idxs[-self.chunk_size :]
                    if use_only_positive_chunk and self.mode == "train":
                        if np.sum(self.labels[idxs]) > 0:
                            indices.append(idxs)
                    elif use_only_non_periodicity and self.mode == "train":
                        if (self.cat_features[idxs, periodicity_idx] == 2).sum() / len(idxs) <= 0.5:
                            indices.append(idxs)
                    else:
                        indices.append(idxs)
            else:
                if use_only_positive_chunk and self.mode == "train":
                    if np.sum(self.labels[-self.chunk_size :]) > 0:
                        indices.append(row_idxs)
                elif use_only_non_periodicity and self.mode == "train":
                    if self.cat_features[-self.chunk_size :, periodicity_idx].sum() == len(
                        row_idxs
                    ):
                        indices.append(row_idxs)
                else:
                    indices.append(row_idxs)

        return indices

    def set_refinement_step(self):
        self.refinement_step = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        indices = self.indices[idx]
        ids = self.ids[indices].tolist()
        steps = self.steps[indices]
        numerical_features = self.numerical_features[indices]
        cat_features = self.cat_features[indices]
        seq_len = len(numerical_features)
        attention_mask = np.ones(seq_len)
        if seq_len < self.chunk_size:
            numerical_features = np.pad(
                numerical_features, ((0, self.chunk_size - seq_len), (0, 0))
            )
            cat_features = np.pad(cat_features, ((0, self.chunk_size - seq_len), (0, 0)))
            steps = np.pad(steps, (0, self.chunk_size - seq_len), constant_values=-1)
            attention_mask = np.pad(
                attention_mask, (0, self.chunk_size - seq_len), constant_values=0
            )
        if self.config.downsample_feature:
            numerical_features = numerical_features.reshape(
                len(numerical_features) // 12, 12, -1
            ).mean(axis=1, dtype=np.float32)
            cat_features = cat_features.reshape(len(cat_features) // 12, 12, -1).mean(
                axis=1, dtype=np.int64
            )
            attention_mask = attention_mask.reshape(len(attention_mask) // 12, 12).mean(
                axis=1, dtype=np.int64
            )

        out = {
            self.id_col: ids[0],
            "numerical_feature": numerical_features.astype(np.float32),
            "cat_feature": cat_features.astype(np.float32),
            "step": steps.astype(np.float32),
            "attention_mask": attention_mask,
        }
        # if self.normalize:
        #     std = StandardScaler()
        #     out["numerical_feature"] = std.fit_transform(out["numerical_feature"])
        if self.mode != "test":
            labels = self.labels[indices]
            if seq_len < self.chunk_size:
                if labels.shape[1] == 1:
                    labels = np.pad(labels, (0, self.chunk_size - seq_len), constant_values=-1)
                else:
                    labels = np.pad(
                        labels, ((0, self.chunk_size - seq_len), (0, 0)), constant_values=-1
                    )
            out.update({self.label_col: labels})
        return out


class StackingDataset(object):
    def __init__(self, config: DictConfig, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load
        logger.info("start load data")
        train_df = pd.read_pickle(f"./input/stacking/{config.feature.version}.pkl")
        train_df = train_df.filter(pl.col("timestamp").is_not_null())
        train_df = train_df.with_columns(pl.Series("row_idx", np.arange(len(train_df))))
        train_df = train_df.with_columns(
            (pl.col("event") == "onset").cast(pl.Int64).fill_null(0).alias("label_onset"),
            (pl.col("event") == "wakeup").cast(pl.Int64).fill_null(0).alias("label_wakeup"),
        )
        features = train_df[config.feature.numerical_cols + config.feature.cat_cols].to_numpy()

        self.id_col = config.id_col
        self.label_col = config.label_col
        self.label_cols = config.feature.label_cols
        # preprocessing
        logger.info("start collect")
        if mode == "train":
            df = train_df.filter(pl.col("fold") != config.n_fold)
        elif mode == "valid":
            df = train_df.filter(pl.col("fold") == config.n_fold)
        self.labels = df.select(self.label_cols).to_numpy()
        self.numerical_features = features[
            df.select("row_idx").to_numpy().flatten(),
            : len(config.feature.numerical_cols),
        ]
        for i in range(self.numerical_features.shape[1]):
            self.numerical_features[:, i] = np.nan_to_num(
                self.numerical_features[:, i], nan=self.numerical_features[:, i].mean()
            )
        self.cat_features = features[
            df.select("row_idx").to_numpy().flatten(),
            len(config.feature.numerical_cols) :,
        ]

        self.series_step_timestamps = df[["series_id", "step", "timestamp"]].unique()

        id_df = df.select([self.id_col, "chunk_id"])
        row_idxs_list = (
            id_df.with_columns(pl.Series("row_idx", np.arange(len(id_df))))
            .group_by(["series_id", "chunk_id"], maintain_order=True)
            .agg(pl.col("row_idx"))["row_idx"]
            .to_numpy()
        )
        self.ids = id_df[self.id_col].to_numpy().flatten()
        logger.info("start normalize")
        if config.normalize:
            if mode == "train":
                std = StandardScaler()
                self.numerical_features = std.fit_transform(self.numerical_features)
                with open(config.std_path, "wb") as f:
                    pickle.dump(std, f)
            else:
                with open(config.std_path, "rb") as f:
                    std = pickle.load(f)
                self.numerical_features = std.transform(self.numerical_features)
        logger.info("finish normalize")

        self.steps = df.select("step").to_numpy().flatten()
        self.numerial_cols = config.feature.numerical_cols
        self.window_size = config.window_size
        self.chunk_size = config.chunk_size
        self.normalize = config.normalize

        logger.info("start create indices")
        self.indices = self._create_indices(row_idxs_list, self.config.use_only_positive_chunk)
        logger.info("finish create indices")

    def _create_indices(
        self,
        row_idxs_list,
        use_only_positive_chunk: bool = False,
    ):
        indices = []
        for row_idxs in row_idxs_list:
            if len(row_idxs) > self.chunk_size:
                for i in range(0, len(row_idxs), self.window_size):
                    idxs = row_idxs[i : i + self.chunk_size]
                    if use_only_positive_chunk and self.mode == "train":
                        if np.sum(self.labels[idxs]) > 0:
                            indices.append(idxs)
                    else:
                        indices.append(idxs)
            else:
                if use_only_positive_chunk and self.mode == "train":
                    if np.sum(self.labels[row_idxs]) > 0:
                        indices.append(row_idxs)
                else:
                    indices.append(row_idxs)

        return indices

    def set_refinement_step(self):
        self.refinement_step = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        indices = self.indices[idx]
        ids = self.ids[indices].tolist()
        steps = self.steps[indices]
        numerical_features = self.numerical_features[indices]
        cat_features = self.cat_features[indices]
        seq_len = len(numerical_features)
        attention_mask = np.ones(seq_len)
        if seq_len < self.chunk_size:
            numerical_features = np.pad(
                numerical_features, ((0, self.chunk_size - seq_len), (0, 0))
            )
            cat_features = np.pad(cat_features, ((0, self.chunk_size - seq_len), (0, 0)))
            steps = np.pad(steps, (0, self.chunk_size - seq_len), constant_values=-1)
            attention_mask = np.pad(
                attention_mask, (0, self.chunk_size - seq_len), constant_values=0
            )
        if self.config.downsample_feature:
            numerical_features = numerical_features.reshape(
                len(numerical_features) // 12, 12, -1
            ).mean(axis=1, dtype=np.float32)
            cat_features = cat_features.reshape(len(cat_features) // 12, 12, -1).mean(
                axis=1, dtype=np.int64
            )
            attention_mask = attention_mask.reshape(len(attention_mask) // 12, 12).mean(
                axis=1, dtype=np.int64
            )

        out = {
            self.id_col: ids[0],
            "numerical_feature": numerical_features.astype(np.float32),
            "cat_feature": cat_features.astype(np.float32),
            "step": steps.astype(np.float32),
            "attention_mask": attention_mask,
        }
        # if self.normalize:
        #     std = StandardScaler()
        #     out["numerical_feature"] = std.fit_transform(out["numerical_feature"])
        if self.mode != "test":
            labels = self.labels[indices]
            if seq_len < self.chunk_size:
                if labels.shape[1] == 1:
                    labels = np.pad(labels, (0, self.chunk_size - seq_len), constant_values=-1)
                else:
                    labels = np.pad(
                        labels, ((0, self.chunk_size - seq_len), (0, 0)), constant_values=-1
                    )
            out.update({self.label_col: labels})
        return out


def get_dataset(config, mode):
    print("dataset class:", config.dataset_class)
    f = globals().get(config.dataset_class)
    return f(config, mode)
