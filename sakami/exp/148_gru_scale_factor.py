from __future__ import annotations

import gc
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import meta_config
from src.metric import score
from src.utils import clear_memory, freeze, set_seed, timer


@freeze
class config:
    chunk_size = int(12 * 60 * 24 * 1)
    stride_size = int(12 * 60 * 24 * 0.7)
    epoch_sample_rate = 2

    preds_truncate_size = 12 * 30

    fold_dir = meta_config.input_dir / "folds"
    folds = [0, 1, 2, 3, 4]

    n_epochs = 10
    warmup_rate = 0.1
    warmup_lr_init = 4e-4
    lr = 8e-4
    lr_min = 1e-6

    batch_size = 4
    eval_batch_size = 64

    height = 0.001
    distance = 110
    daily_score_offset = 0.4

    periodicity_dir = meta_config.input_dir / "periodicity"
    filter_size = 10

    prune_epoch = 3
    prune_score = 0.1

    device = torch.device("cuda")
    seed = 1029


class SleepDataset(Dataset):
    def __init__(
        self,
        series_chunk_row_ids: list[np.ndarray],
        numerical_inputs: np.ndarray,
        categorical_inputs: np.ndarray,
        targets: np.ndarray,
        max_series_length: int,
        indices: np.ndarray | None = None,
    ) -> None:
        self.series_chunk_row_ids = series_chunk_row_ids

        self.numerical_inputs = numerical_inputs
        self.categorical_inputs = categorical_inputs
        self.targets = targets

        self.max_series_length = max_series_length

        if indices is None:
            indices = np.arange(len(self.series_chunk_row_ids))

        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        index = self.indices[idx]
        series_row_ids = self.series_chunk_row_ids[index]

        num_x = self.numerical_inputs[series_row_ids]  # (sequence_length, num_numerical_features)
        cat_x = self.categorical_inputs[series_row_ids]  # (sequence_length, num_categorical_features)
        attention_mask = (series_row_ids != -1).astype(np.float32)  # (sequence_length,)
        target = self.targets[series_row_ids]  # (sequence_length, num_targets)

        assert (
            self.max_series_length
            == len(series_row_ids)
            == len(num_x)
            == len(cat_x)
            == len(attention_mask)
            == len(target)
        )
        return series_row_ids, num_x, cat_x, attention_mask, target


class SEScale(nn.Module):
    def __init__(self, ch: int, r: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(ch, r)
        self.fc2 = nn.Linear(r, ch)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h).sigmoid()
        return h * x


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int = 1, bidirectional: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        dir_factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * dir_factor, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.PReLU(),
        )

    def forward(self, x, h=None):
        res, _ = self.gru(x, h)
        res = self.fc(res)
        res = res + x
        return res, _


class SleepModel(nn.Module):
    def __init__(self, numerical_input_size: int, categorical_input_size: int) -> None:
        super().__init__()
        self.embedding_size = 32
        self.hidden_size = 64
        self.n_layers = 8
        self.dropout = 0.1
        self.kernel_sizes = [11, 7, 5]
        self.stride = 2

        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(128, self.embedding_size, padding_idx=0) for _ in range(categorical_input_size)]
        )
        self.numerical_linear = nn.Sequential(
            SEScale(numerical_input_size, 8),
            nn.Linear(numerical_input_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.PReLU(),
        )
        input_size = self.embedding_size * categorical_input_size + self.hidden_size
        self.input_linear = nn.Sequential(
            SEScale(input_size, 64),
            nn.Linear(input_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.PReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1, padding="same"),
            nn.Dropout(self.dropout),
            nn.PReLU(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        padding=kernel_size // 2,
                    ),
                    nn.Dropout(self.dropout),
                    nn.PReLU(),
                    nn.Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                    ),
                    nn.Dropout(self.dropout),
                    nn.PReLU(),
                )
                for kernel_size in self.kernel_sizes
            ],
        )

        self.gru_layers = nn.ModuleList(
            [ResidualGRU(self.hidden_size, n_layers=1, bidirectional=True) for _ in range(self.n_layers)]
        )

        self.dconv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        padding=kernel_size // 2,
                        output_padding=1,
                    ),
                    nn.Dropout(self.dropout),
                    nn.PReLU(),
                    nn.Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                    ),
                    nn.Dropout(self.dropout),
                    nn.PReLU(),
                )
                for kernel_size in reversed(self.kernel_sizes)
            ]
        )

        self.output_linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.hidden_size, 32),
            nn.Dropout(self.dropout),
            nn.PReLU(),
            nn.Linear(32, 2),
        )
        self.output_scale_factor = nn.Parameter(torch.zeros(1, dtype=torch.float))

    def forward(self, num_x: torch.FloatTensor, cat_x: torch.LongTensor) -> torch.FloatTensor:
        cat_embeddings = [embedding(cat_x[:, :, i]) for i, embedding in enumerate(self.category_embeddings)]
        num_x = self.numerical_linear(num_x)

        x = torch.cat([num_x] + cat_embeddings, dim=2)
        x = self.input_linear(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)

        for gru in self.gru_layers:
            x, _ = gru(x)

        x = self.dconv(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.output_linear(x)
        return x * self.output_scale_factor.exp()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def predict(model: nn.Module, data_loader: DataLoader, preds_truncate_size: int) -> pl.DataFrame:
    model.eval()
    device = model.device

    row_ids = []
    preds = []

    with torch.inference_mode():
        for series_row_ids, num_x, cat_x, attention_mask, _ in tqdm(data_loader, desc="predict", leave=False):
            series_row_ids = series_row_ids[:, preds_truncate_size:-preds_truncate_size]
            attention_mask = attention_mask[:, preds_truncate_size:-preds_truncate_size]
            series_row_ids = series_row_ids.masked_select(attention_mask.bool()).numpy()
            row_ids.append(series_row_ids)

            pred = (
                model(num_x.to(device), cat_x.to(device))[:, preds_truncate_size:-preds_truncate_size]
                .detach()
                .sigmoid()
                .cpu()
            )  # (batch_size, sequence_length - preds_truncate_size * 2, 2)
            pred_mask = attention_mask.unsqueeze(-1).expand(-1, -1, 2).bool()
            pred = pred.masked_select(pred_mask).view(-1, 2).numpy()
            preds.append(pred)

    # aggregate overlapping predictions
    preds_df = pl.DataFrame(
        {
            "row_id": np.concatenate(row_ids, axis=0),
            "prediction_onset": np.concatenate(preds, axis=0)[:, 0],
            "prediction_wakeup": np.concatenate(preds, axis=0)[:, 1],
        }
    )
    preds_df = preds_df.group_by("row_id").mean().sort("row_id")
    return preds_df


def make_submission(
    preds_df: pl.DataFrame,
    periodicity_dict: dict[str, np.ndarray],
    height: float = 0.001,
    distance: int = 100,
    daily_score_offset: float = 1.0,
) -> pl.DataFrame:
    event_dfs = []

    for series_id, series_df in tqdm(
        preds_df.group_by("series_id"), desc="find peaks", leave=False, total=preds_df["series_id"].n_unique()
    ):
        for event in ["onset", "wakeup"]:
            event_preds = series_df[f"prediction_{event}"].to_numpy().copy()
            event_preds *= 1 - periodicity_dict[series_id][: len(event_preds)]
            steps = find_peaks(event_preds, height=height, distance=distance)[0]

            event_dfs.append(
                series_df[steps]
                .with_columns(pl.lit(event).alias("event"))
                .rename({f"prediction_{event}": "score"})
                .select(["series_id", "step", "timestamp", "event", "score"])
            )

    submission_df = (
        pl.concat(event_dfs)
        .with_columns(pl.col("timestamp").dt.offset_by("2h").dt.date().alias("date"))
        .with_columns(
            pl.col("score") / (pl.col("score").sum().over(["series_id", "event", "date"]) + daily_score_offset)
        )
        .sort(["series_id", "step"])
        .with_columns(pl.arange(0, pl.count()).alias("row_id"))
        .select(["row_id", "series_id", "step", "event", "score"])
    )
    return submission_df


@clear_memory
def main(debug: bool = False) -> None:
    start_time = time.time()

    # load data
    with timer("load data"):
        train_df = pl.read_parquet(meta_config.data_dir / "train_series.parquet")
        train_events_df = pl.read_csv(meta_config.data_dir / "train_events.csv")

        split_series_ids = []
        for fold in config.folds:
            with open(config.fold_dir / f"stratify_fold_{fold}.yaml", "r") as f:
                split_series_ids.append(yaml.safe_load(f))

    if debug:
        sampled_series_ids = train_df.get_column("series_id").unique().sample(10, seed=config.seed)
        train_df = (
            train_df.filter(pl.col("series_id").is_in(sampled_series_ids)).group_by("series_id").head(12 * 60 * 24)
        )
        train_events_df = train_events_df.filter(pl.col("series_id").is_in(sampled_series_ids))

        split_series_count = 7
        split_series_ids = [
            {
                "train_series_ids": sampled_series_ids[:split_series_count].to_list(),
                "valid_series_ids": sampled_series_ids[split_series_count:].to_list(),
            }
        ]

    # preprocess
    with timer("preprocess"):
        # target columns
        logger.info("\t add target columns")
        tolerance_steps = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
        target_columns = ["event_onset", "event_wakeup"]

        train_events_df = train_events_df.with_columns(pl.col("step").cast(pl.UInt32))
        train_df = (
            train_df.join(train_events_df.select(["series_id", "step", "event"]), on=["series_id", "step"], how="left")
            .to_dummies(columns=["event"])
            .with_columns(
                [
                    pl.max_horizontal(
                        pl.col(target_column)
                        .rolling_max(window_size * 2 - 1, min_periods=1, center=True)
                        .over("series_id")
                        * (1 - 1 / len(tolerance_steps) * i)
                        for i, window_size in enumerate(tolerance_steps)
                    ).alias(target_column)
                    for target_column in target_columns
                ]
            )
        )

        # feature engineering
        logger.info("\t feature engineering")
        train_df = train_df.with_columns(
            pl.col("anglez") / 45,
            (pl.col("enmo").log1p() / 0.1).clip_max(10.0),
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%z"),
        ).with_columns(
            pl.col("timestamp").dt.hour().alias("hour") + 1,
            pl.col("timestamp").dt.minute().alias("minute") + 1,
            pl.col("timestamp").dt.weekday().alias("weekday"),
            pl.col(["anglez", "enmo"]).rolling_mean(12, center=True).over("series_id").suffix("_12_rolling_mean"),
            pl.col(["anglez", "enmo"]).rolling_std(12, center=True).over("series_id").suffix("_12_rolling_std"),
            pl.col(["anglez", "enmo"]).rolling_max(12, center=True).over("series_id").suffix("_12_rolling_max"),
            pl.col("anglez").diff().abs().rolling_median(60, center=True).over("series_id").suffix("_diff_5min_median"),
        )

        # add periodicity
        periodicity_series_ids = []
        periodicity_flags = []
        periodicity_steps = []
        periodicity_dict = {}
        all_series_ids = train_df["series_id"].unique().to_numpy()

        for series_id in all_series_ids:
            periodicity_flag = np.load(config.periodicity_dir / series_id / "periodicity.npy")
            periodicity = np.minimum(periodicity_flag, uniform_filter1d(periodicity_flag, size=config.filter_size))

            periodicity_series_ids.append(series_id)
            periodicity_flags.append(periodicity_flag)
            periodicity_steps.append(np.arange(len(periodicity_flag)))

            periodicity_dict[series_id] = periodicity

        periodicity_df = (
            pl.DataFrame(
                {
                    "series_id": periodicity_series_ids,
                    "step": periodicity_steps,
                    "periodicity": periodicity_flags,
                }
            )
            .explode(columns=["step", "periodicity"])
            .with_columns(pl.col("step").cast(pl.UInt32), pl.col("periodicity") + 1)
        )
        train_df = train_df.join(periodicity_df, how="left", on=["series_id", "step"])

        categorical_columns = ["hour", "minute", "weekday", "periodicity"]
        numerical_columns = train_df.select(r"^(anglez|enmo).*$").columns

        # split into chunks
        logger.info("\t split into chunks")
        train_df = train_df.with_columns(pl.arange(0, pl.count()).alias("row_id"))
        series_row_ids = dict(train_df.group_by("series_id").agg("row_id").rows())

        series_chunk_ids = []  # list[str]
        series_chunk_row_ids = []  # list[list[int]]
        for series_id, row_ids in tqdm(series_row_ids.items(), desc="split into chunks"):
            for start_idx in range(
                -config.preds_truncate_size, len(row_ids), int(config.stride_size / config.epoch_sample_rate)
            ):
                if start_idx + config.chunk_size <= len(row_ids) + config.preds_truncate_size:
                    chunk_row_ids = row_ids[max(0, start_idx) : start_idx + config.chunk_size]

                    # padding
                    if len(chunk_row_ids) < config.chunk_size:
                        chunk_row_ids = [-1] * (config.chunk_size - len(chunk_row_ids)) + chunk_row_ids

                    series_chunk_ids.append(series_id)
                    series_chunk_row_ids.append(np.array(chunk_row_ids))
                else:
                    chunk_row_ids = row_ids[-config.chunk_size + config.preds_truncate_size :]

                    # padding
                    if len(chunk_row_ids) < config.chunk_size:
                        chunk_row_ids = chunk_row_ids + [-1] * (config.chunk_size - len(chunk_row_ids))

                    series_chunk_ids.append(series_id)
                    series_chunk_row_ids.append(np.array(chunk_row_ids))
                    break

        # drop features
        numerical_inputs = train_df.select(numerical_columns).fill_null(0.0).to_numpy().astype(np.float32)
        categorical_inputs = train_df.select(categorical_columns).fill_null(0.0).to_numpy().astype(np.int64)
        target_columns = train_df.select(r"^event_(onset|wakeup).*$").columns
        train_df = train_df.select(["row_id", "series_id", "step", "timestamp", "periodicity"] + target_columns)

        # pad inputs
        numerical_inputs = np.pad(numerical_inputs, ((0, 1), (0, 0)))
        categorical_inputs = np.pad(categorical_inputs, ((0, 1), (0, 0)))

        # split index
        series_chunk_ids = np.array(series_chunk_ids)
        splits = [
            (
                np.where(np.isin(series_chunk_ids, split_series_id["train_series_ids"]))[0],  # train_idx
                np.where(np.isin(series_chunk_ids, split_series_id["valid_series_ids"]))[0],  # valid_idx
            )
            for split_series_id in split_series_ids
        ]

    logger.info("")
    logger.info(f"train shape : {train_df.shape}")
    logger.info(f"train events shape : {train_events_df.shape}")
    logger.info(f"train series count : {len(series_row_ids)}")
    logger.info(f"train series chunk count : {len(series_chunk_row_ids)}")

    logger.info("")
    logger.info("categorical columns:")
    for column in categorical_columns:
        logger.info(f"\t {column}")

    logger.info("numerical columns:")
    for column in numerical_columns:
        logger.info(f"\t {column}")
    logger.info("")

    # train
    with timer("train"):
        valid_preds = []
        valid_submissions = []
        cv_scores = []

        model_checkpoint_dir = meta_config.checkpoint_dir / "models"
        data_checkpoint_dir = meta_config.checkpoint_dir / "data"

        model_checkpoint_dir.mkdir(exist_ok=True)
        data_checkpoint_dir.mkdir(exist_ok=True)

        for fold, (train_idx, valid_idx) in zip(config.folds, splits):
            logger.info("-" * 40)
            logger.info(f"fold {fold}")

            model_path = model_checkpoint_dir / f"model_fold{fold}.pth"

            targets = train_df.select(target_columns).to_numpy().astype(np.float32)
            targets = np.pad(targets, ((0, 1), (0, 0)))
            valid_series_ids = set(np.array(series_chunk_ids)[valid_idx])
            valid_df = train_df.filter(pl.col("series_id").is_in(valid_series_ids))
            valid_events_df = train_events_df.filter(pl.col("series_id").is_in(valid_series_ids))

            model = SleepModel(
                numerical_input_size=len(numerical_columns),
                categorical_input_size=len(categorical_columns),
            )
            model.zero_grad()
            model.to(config.device)

            # drop no event sequence
            all_train_idx = []
            for index in train_idx:
                row_ids = series_chunk_row_ids[index]
                if train_df[row_ids, "periodicity"].to_numpy().min() != 2:
                    all_train_idx.append(index)
            train_idx = np.array(all_train_idx)

            train_size = math.ceil(len(train_idx) / config.epoch_sample_rate / config.batch_size)
            t_initial = train_size * config.n_epochs
            warmup_t = int(t_initial * config.warmup_rate)

            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=t_initial,
                lr_min=config.lr_min,
                warmup_t=warmup_t,
                warmup_lr_init=config.warmup_lr_init,
            )

            loss_ema = None
            best_epoch = None
            best_score = -np.inf
            best_valid_fold_preds_df = None
            best_valid_fold_submission = None

            for epoch in range(config.n_epochs):
                epoch_start_time = time.time()
                model.train()

                sampled_train_idx = train_idx[epoch % config.epoch_sample_rate :: config.epoch_sample_rate]

                train_dataset = SleepDataset(
                    series_chunk_row_ids=series_chunk_row_ids,
                    numerical_inputs=numerical_inputs,
                    categorical_inputs=categorical_inputs,
                    targets=targets,
                    max_series_length=config.chunk_size,
                    indices=sampled_train_idx,
                )
                valid_dataset = SleepDataset(
                    series_chunk_row_ids=series_chunk_row_ids,
                    numerical_inputs=numerical_inputs,
                    categorical_inputs=categorical_inputs,
                    targets=targets,
                    max_series_length=config.chunk_size,
                    indices=valid_idx,
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=4,
                    pin_memory=True,
                )
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=config.eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=True,
                )

                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
                for step, (_, num_x, cat_x, attention_mask, target) in enumerate(progress):
                    preds = model(num_x.to(config.device), cat_x.to(config.device))
                    loss = (
                        nn.BCEWithLogitsLoss(reduction="none")(preds, target.to(config.device))
                        * attention_mask.unsqueeze(-1).to(config.device)
                    ).mean()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step(step + train_size * epoch)

                    loss_ema = loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                    progress.set_postfix(loss=loss_ema)

                epoch_elapsed_time = (time.time() - epoch_start_time) / 60
                predict_start_time = time.time()

                valid_fold_preds_df = predict(model, valid_loader, preds_truncate_size=config.preds_truncate_size)
                valid_fold_preds_df = (
                    valid_df[["row_id"]].join(valid_fold_preds_df, how="left", on="row_id").fill_null(0.0)
                )

                valid_preds_df = valid_df.with_columns(
                    valid_fold_preds_df.select(["prediction_onset", "prediction_wakeup"])
                )
                valid_fold_submission = make_submission(
                    valid_preds_df,
                    periodicity_dict,
                    height=config.height,
                    distance=config.distance,
                    daily_score_offset=config.daily_score_offset,
                )
                valid_score = score(valid_events_df, valid_fold_submission)

                predict_elapsed_time = (time.time() - predict_start_time) / 60
                logger.info(
                    f"  Epoch {epoch + 1}"
                    f" \t train loss: {loss_ema:.5f}"
                    f" \t valid score: {valid_score:.5f}"
                    f" \t train time: {epoch_elapsed_time:.2f} min"
                    f" \t predict time: {predict_elapsed_time:.2f} min"
                )

                if valid_score > best_score:
                    best_epoch = epoch
                    best_score = valid_score
                    best_valid_fold_preds_df = valid_fold_preds_df
                    best_valid_fold_submission = valid_fold_submission
                    torch.save(model.state_dict(), model_path)

                if epoch >= config.prune_epoch and valid_score < config.prune_score:
                    raise Exception(f"Training failed. epoch: {epoch + 1}, score: {valid_score:.5f}")

                if debug:
                    break

                # update target
                targets = np.where(targets == 1.0, 1.0, (targets - 0.1).clip(min=0.0))

            logger.info(f"* Best epoch: {best_epoch + 1} \t valid score: {best_score:.5f}")
            cv_scores.append(best_score)
            valid_preds.append(best_valid_fold_preds_df)
            valid_submissions.append(best_valid_fold_submission)

            # save
            best_valid_fold_preds_df.write_parquet(data_checkpoint_dir / f"valid_fold{fold}_preds.parquet")
            best_valid_fold_submission.write_parquet(data_checkpoint_dir / f"valid_fold{fold}_submission.parquet")

            valid_fold_preds_df.write_parquet(data_checkpoint_dir / f"last_valid_fold{fold}_preds.parquet")
            valid_fold_submission.write_parquet(data_checkpoint_dir / f"last_valid_fold{fold}_submission.parquet")

            del train_dataset, valid_dataset
            del train_loader, valid_loader
            del valid_df, valid_events_df
            del valid_preds_df, valid_fold_submission
            del best_valid_fold_preds_df, best_valid_fold_submission
            del model, optimizer, scheduler
            gc.collect()

            if debug:
                break

    cv_score = np.mean(cv_scores)
    logger.info(f"cv score: {cv_score:.5f}")

    # save
    pl.concat(valid_preds).sort("row_id").write_parquet(meta_config.save_dir / "valid_preds.parquet")
    pl.concat(valid_submissions).with_columns(pl.arange(0, pl.count()).alias("row_id")).sort("row_id").write_parquet(
        meta_config.save_dir / "valid_submission.parquet"
    )

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"all processes done in {elapsed_time:.1f} min.")


def run() -> None:
    # debug
    logger.opt(colors=True).info("<yellow>********************** mode : debug **********************</yellow>")
    main(debug=True)

    # main
    logger.opt(colors=True).info("<yellow>" + "-" * 60 + "</yellow>")
    logger.opt(colors=True).info("<yellow>********************** mode : main **********************</yellow>")
    set_seed(config.seed)
    main(debug=False)


if __name__ == "__main__":
    run()
