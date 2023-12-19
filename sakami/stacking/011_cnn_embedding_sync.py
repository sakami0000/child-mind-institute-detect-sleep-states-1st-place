from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from src import meta_config
from src.detect_peak import post_process_from_2nd
from src.metric import score
from src.utils import clear_memory, freeze, set_seed, timer
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@freeze
class config:
    input_dir = meta_config.input_dir / "stacking/"
    input_file = "009.parquet"

    fold_dir = meta_config.input_dir / "folds"
    folds = [0, 1, 2, 3, 4]

    n_epochs = 12
    warmup_rate = 0.1
    warmup_lr_init = 5e-4
    lr = 1e-3
    lr_min = 1e-6

    batch_size = 16
    eval_batch_size = 64

    event_rate = 500
    height = 0.001
    onset_date_offset = "5h"
    wakeup_date_offset = "0h"
    use_daily_norm = True
    daily_score_offset = 5
    later_date_max_sub_rate = None

    periodicity_dir = Path("../kami/processed/train")
    filter_size = 10

    prune_epoch = 3
    prune_score = 0.0

    device = torch.device("cuda")
    seed = 1029

    numerical_columns = [
        "pred_onset_148_gru_scale_factor",
        "pred_onset_156_gru_transformer_residual",
        "pred_onset_163_gru_sleep_target",
        "pred_onset_exp068_transformer",
        "pred_onset_exp078_lstm",
        "pred_wakeup_148_gru_scale_factor",
        "pred_wakeup_156_gru_transformer_residual",
        "pred_wakeup_163_gru_sleep_target",
        "pred_wakeup_exp068_transformer",
        "pred_wakeup_exp078_lstm",
        "pred_sleep_exp068_transformer",
        "pred_sleep_exp078_lstm",
        "pred_sleep_163_gru_sleep_target",
        # "anglez",
        # "enmo",
        "prediction_onset",
        # "prediction_onset_min",
        # "prediction_onset_max",
        "prediction_wakeup",
        # "prediction_wakeup_min",
        # "prediction_wakeup_max",
        # "periodicity",
        # "activity_count",
        # "lid",
        # "enmo_3_rolling_mean",
        # "enmo_3_rolling_std",
        # "enmo_3_rolling_max",
        # "anglez_3_rolling_mean",
        # "anglez_3_rolling_std",
        # "anglez_3_rolling_max",
        # "lid_3_rolling_mean",
        # "lid_3_rolling_std",
        # "lid_3_rolling_max",
        # "enmo_diff_prev3_rolling_mean",
        # "anglez_diff_prev3_rolling_mean",
        # "lid_diff_prev3_rolling_mean",
        # "enmo_diff_lead3_rolling_mean",
        # "anglez_diff_lead3_rolling_mean",
        # "lid_diff_lead3_rolling_mean",
        # "enmo_diff_lag",
        # "anglez_diff_lag",
        # "lid_diff_lag",
        # "pred_onset_148_gru_scale_factor_diff_prev3_rolling_mean",
        # "pred_onset_156_gru_transformer_residual_diff_prev3_rolling_mean",
        # "pred_onset_163_gru_sleep_target_diff_prev3_rolling_mean",
        # "pred_onset_exp068_transformer_diff_prev3_rolling_mean",
        # "pred_onset_exp078_lstm_diff_prev3_rolling_mean",
        # "pred_wakeup_148_gru_scale_factor_diff_prev3_rolling_mean",
        # "pred_wakeup_156_gru_transformer_residual_diff_prev3_rolling_mean",
        # "pred_wakeup_163_gru_sleep_target_diff_prev3_rolling_mean",
        # "pred_wakeup_exp068_transformer_diff_prev3_rolling_mean",
        # "pred_wakeup_exp078_lstm_diff_prev3_rolling_mean",
        # "pred_onset_148_gru_scale_factor_diff_prev5_rolling_mean",
        # "pred_onset_156_gru_transformer_residual_diff_prev5_rolling_mean",
        # "pred_onset_163_gru_sleep_target_diff_prev5_rolling_mean",
        # "pred_onset_exp068_transformer_diff_prev5_rolling_mean",
        # "pred_onset_exp078_lstm_diff_prev5_rolling_mean",
        # "pred_wakeup_148_gru_scale_factor_diff_prev5_rolling_mean",
        # "pred_wakeup_156_gru_transformer_residual_diff_prev5_rolling_mean",
        # "pred_wakeup_163_gru_sleep_target_diff_prev5_rolling_mean",
        # "pred_wakeup_exp068_transformer_diff_prev5_rolling_mean",
        # "pred_wakeup_exp078_lstm_diff_prev5_rolling_mean",
        # "pred_onset_148_gru_scale_factor_diff_lead3_rolling_mean",
        # "pred_onset_156_gru_transformer_residual_diff_lead3_rolling_mean",
        # "pred_onset_163_gru_sleep_target_diff_lead3_rolling_mean",
        # "pred_onset_exp068_transformer_diff_lead3_rolling_mean",
        # "pred_onset_exp078_lstm_diff_lead3_rolling_mean",
        # "pred_wakeup_148_gru_scale_factor_diff_lead3_rolling_mean",
        # "pred_wakeup_156_gru_transformer_residual_diff_lead3_rolling_mean",
        # "pred_wakeup_163_gru_sleep_target_diff_lead3_rolling_mean",
        # "pred_wakeup_exp068_transformer_diff_lead3_rolling_mean",
        # "pred_wakeup_exp078_lstm_diff_lead3_rolling_mean",
        # "pred_onset_148_gru_scale_factor_diff_lead5_rolling_mean",
        # "pred_onset_156_gru_transformer_residual_diff_lead5_rolling_mean",
        # "pred_onset_163_gru_sleep_target_diff_lead5_rolling_mean",
        # "pred_onset_exp068_transformer_diff_lead5_rolling_mean",
        # "pred_onset_exp078_lstm_diff_lead5_rolling_mean",
        # "pred_wakeup_148_gru_scale_factor_diff_lead5_rolling_mean",
        # "pred_wakeup_156_gru_transformer_residual_diff_lead5_rolling_mean",
        # "pred_wakeup_163_gru_sleep_target_diff_lead5_rolling_mean",
        # "pred_wakeup_exp068_transformer_diff_lead5_rolling_mean",
        # "pred_wakeup_exp078_lstm_diff_lead5_rolling_mean",
        # "pred_onset_148_gru_scale_factor_diff_lag",
        # "pred_onset_156_gru_transformer_residual_diff_lag",
        # "pred_onset_163_gru_sleep_target_diff_lag",
        # "pred_onset_exp068_transformer_diff_lag",
        # "pred_onset_exp078_lstm_diff_lag",
        # "pred_wakeup_148_gru_scale_factor_diff_lag",
        # "pred_wakeup_156_gru_transformer_residual_diff_lag",
        # "pred_wakeup_163_gru_sleep_target_diff_lag",
        # "pred_wakeup_exp068_transformer_diff_lag",
        # "pred_wakeup_exp078_lstm_diff_lag",
    ]
    categorical_columns = [
        "minute",
        # "time_idx",
        "hour",
        # "minute_mod15",
        "weekday",
        "periodicity",
    ]


class SleepDataset(Dataset):
    def __init__(
        self,
        series_chunk_row_ids: list[np.ndarray],
        numerical_inputs: np.ndarray,
        categorical_inputs: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> None:
        self.series_chunk_row_ids = series_chunk_row_ids

        self.numerical_inputs = numerical_inputs
        self.categorical_inputs = categorical_inputs
        self.targets = targets

        if indices is None:
            indices = np.arange(len(self.series_chunk_row_ids))

        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        index = self.indices[idx]
        series_row_ids = self.series_chunk_row_ids[index]

        num_x = self.numerical_inputs[series_row_ids]  # (sequence_length, num_numerical_features)
        cat_x = self.categorical_inputs[
            series_row_ids
        ]  # (sequence_length, num_categorical_features)
        target = self.targets[series_row_ids]  # (sequence_length, num_targets)

        assert len(series_row_ids) == len(num_x) == len(cat_x) == len(target)
        return series_row_ids, num_x, cat_x, target


def pad_sequence(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[
    torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor
]:
    batch_series_row_ids, batch_num_x, batch_cat_x, batch_target = zip(*batch)
    max_sequence_length = max(len(series_row_ids) for series_row_ids in batch_series_row_ids)

    padded_series_row_ids = np.full(
        (
            len(batch),
            max_sequence_length,
        ),
        fill_value=-1,
        dtype=np.int64,
    )
    padded_num_x = np.zeros(
        (len(batch), max_sequence_length, batch_num_x[0].shape[-1]), dtype=np.float32
    )
    padded_cat_x = np.zeros(
        (len(batch), max_sequence_length, batch_cat_x[0].shape[-1]), dtype=np.int64
    )
    padded_target = np.zeros(
        (len(batch), max_sequence_length, batch_target[0].shape[-1]), dtype=np.float32
    )
    attention_mask = np.zeros((len(batch), max_sequence_length), dtype=np.float32)

    for i, (series_row_ids, num_x, cat_x, target) in enumerate(batch):
        padded_series_row_ids[i, : len(series_row_ids)] = series_row_ids
        padded_num_x[i, : len(num_x)] = num_x
        padded_cat_x[i, : len(cat_x)] = cat_x
        padded_target[i, : len(target)] = target
        attention_mask[i, : len(series_row_ids)] = 1.0

    series_row_ids = torch.from_numpy(padded_series_row_ids)
    num_x = torch.from_numpy(padded_num_x)
    cat_x = torch.from_numpy(padded_cat_x)
    attention_mask = torch.from_numpy(attention_mask)
    target = torch.from_numpy(padded_target)
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


class SleepModel(nn.Module):
    def __init__(self, numerical_input_size: int, categorical_input_size: int) -> None:
        super().__init__()
        self.embedding_size = 32
        self.hidden_size = 128
        self.n_layers = 8
        self.kernel_size = 3
        self.dropout = 0.1

        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(128, self.embedding_size, padding_idx=0)
                for _ in range(categorical_input_size)
            ]
        )
        self.numerical_linear = nn.Sequential(
            SEScale(numerical_input_size, 8),
            nn.Linear(numerical_input_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.PReLU(),
        )
        input_size = self.embedding_size * categorical_input_size + self.hidden_size
        self.input_linear = nn.Sequential(
            SEScale(input_size, 32),
            nn.Linear(input_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.PReLU(),
        )
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        self.hidden_size,
                        self.hidden_size,
                        kernel_size=self.kernel_size,
                        padding="same",
                    ),
                    nn.Dropout(self.dropout),
                    nn.PReLU(),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.output_linear = nn.Sequential(
            nn.Linear(self.embedding_size + self.hidden_size, 32),
            nn.Dropout(self.dropout),
            nn.PReLU(),
            nn.Linear(32, 2),
        )

    def forward(
        self, num_x: torch.FloatTensor, cat_x: torch.LongTensor, attention_mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        cat_embeddings = [
            embedding(cat_x[:, :, i]) for i, embedding in enumerate(self.category_embeddings)
        ]
        num_x = self.numerical_linear(num_x)

        x = torch.cat([num_x] + cat_embeddings, dim=2)
        x = self.input_linear(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.output_linear(torch.cat([x, cat_embeddings[0]], dim=2))
        return x

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def predict(model: nn.Module, data_loader: DataLoader) -> pl.DataFrame:
    model.eval()
    device = model.device

    row_ids = []
    preds = []

    with torch.inference_mode():
        for series_row_ids, num_x, cat_x, attention_mask, _ in tqdm(
            data_loader, desc="predict", leave=False
        ):
            series_row_ids = series_row_ids.masked_select(attention_mask.bool()).numpy()
            row_ids.append(series_row_ids)

            pred = (
                model(num_x.to(device), cat_x.to(device), attention_mask.to(device))
                .detach()
                .sigmoid()
                .cpu()
            )  # (batch_size, sequence_length, 2)
            pred_mask = attention_mask.unsqueeze(-1).expand(-1, -1, 2).bool()
            pred = pred.masked_select(pred_mask).view(-1, 2).numpy()
            preds.append(pred)

    # aggregate overlapping predictions
    preds_df = pl.DataFrame(
        {
            "row_id": np.concatenate(row_ids, axis=0),
            "stacking_prediction_onset": np.concatenate(preds, axis=0)[:, 0],
            "stacking_prediction_wakeup": np.concatenate(preds, axis=0)[:, 1],
        }
    )
    preds_df = preds_df.group_by("row_id").mean().sort("row_id")
    return preds_df


@clear_memory
def main(debug: bool = False) -> None:
    start_time = time.time()

    # load data
    with timer("load data"):
        train_df = pl.read_parquet(config.input_dir / config.input_file)
        train_events_df = pl.read_csv(meta_config.data_dir / "train_events.csv")

        split_series_ids = []
        for fold in config.folds:
            with open(config.fold_dir / f"stratify_fold_{fold}.yaml", "r") as f:
                split_series_ids.append(yaml.safe_load(f))

    if debug:
        sampled_series_ids = train_df.get_column("series_id").unique().sample(4, seed=config.seed)
        train_df = train_df.filter(pl.col("series_id").is_in(sampled_series_ids))
        train_events_df = train_events_df.filter(pl.col("series_id").is_in(sampled_series_ids))

        split_series_count = 2
        split_series_ids = [
            {
                "train_series_ids": sampled_series_ids[:split_series_count].to_list(),
                "valid_series_ids": sampled_series_ids[split_series_count:].to_list(),
            }
        ]

    # preprocess
    with timer("preprocess"):
        # add target column
        train_df = train_df.to_dummies(columns=["event"])

        # split into chunks
        train_df = train_df.with_columns(pl.arange(0, pl.count()).alias("row_id"))
        series_chunk_ids, _, series_chunk_row_ids = zip(
            *train_df.group_by(["series_id", "chunk_id"]).agg("row_id").rows()
        )
        series_chunk_row_ids = [np.array(series_row_ids) for series_row_ids in series_chunk_row_ids]

        # drop features
        assert config.categorical_columns[0] == "minute"

        numerical_inputs = (
            train_df.select(config.numerical_columns).fill_null(0.0).to_numpy().astype(np.float32)
        )
        categorical_inputs = (
            train_df.select(config.categorical_columns).fill_null(0.0).to_numpy().astype(np.int64)
            + 1
        )
        targets = (
            train_df.select(r"^event_(onset|wakeup).*$")
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32)
        )
        train_df = train_df.select(
            ["row_id", "series_id", "chunk_id", "step", "timestamp", "periodicity"]
        )

        # split index
        series_chunk_ids = np.array(series_chunk_ids)
        splits = [
            (
                np.where(np.isin(series_chunk_ids, split_series_id["train_series_ids"]))[
                    0
                ],  # train_idx
                np.where(np.isin(series_chunk_ids, split_series_id["valid_series_ids"]))[
                    0
                ],  # valid_idx
            )
            for split_series_id in split_series_ids
        ]

    logger.info("")
    logger.info(f"train shape : {train_df.shape}")
    logger.info(f"train events shape : {train_events_df.shape}")
    logger.info(f"train series chunk count : {len(series_chunk_row_ids)}")

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

            valid_series_ids = set(np.array(series_chunk_ids)[valid_idx])
            valid_df = train_df.filter(pl.col("series_id").is_in(valid_series_ids))
            valid_events_df = train_events_df.filter(pl.col("series_id").is_in(valid_series_ids))

            train_dataset = SleepDataset(
                series_chunk_row_ids=series_chunk_row_ids,
                numerical_inputs=numerical_inputs,
                categorical_inputs=categorical_inputs,
                targets=targets,
                indices=train_idx,
            )
            valid_dataset = SleepDataset(
                series_chunk_row_ids=series_chunk_row_ids,
                numerical_inputs=numerical_inputs,
                categorical_inputs=categorical_inputs,
                targets=targets,
                indices=valid_idx,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=pad_sequence,
                num_workers=4,
                pin_memory=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=pad_sequence,
                num_workers=0,
                pin_memory=True,
            )

            model = SleepModel(
                numerical_input_size=len(config.numerical_columns),
                categorical_input_size=len(config.categorical_columns),
            )
            model.zero_grad()
            model.to(config.device)

            train_size = len(train_loader)
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

                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
                for step, (_, num_x, cat_x, attention_mask, target) in enumerate(progress):
                    attention_mask = attention_mask.to(config.device)

                    preds = model(
                        num_x.to(config.device),
                        cat_x.to(config.device),
                        attention_mask=attention_mask,
                    )
                    loss = (
                        nn.BCEWithLogitsLoss(reduction="none")(
                            preds, target.to(config.device).clamp(max=1.0)
                        )
                        * attention_mask.unsqueeze(-1)
                    ).sum() / attention_mask.sum()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step(step + train_size * epoch)

                    loss_ema = (
                        loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                    )
                    progress.set_postfix(loss=loss_ema)

                epoch_elapsed_time = (time.time() - epoch_start_time) / 60
                predict_start_time = time.time()

                valid_fold_preds_df = predict(model, valid_loader)
                valid_preds_df = valid_fold_preds_df.join(
                    valid_df, how="left", on="row_id"
                ).with_columns(
                    pl.col(["stacking_prediction_onset", "stacking_prediction_wakeup"])
                    * (1 - pl.col("periodicity"))
                )

                valid_fold_submission = post_process_from_2nd(
                    valid_preds_df,
                    event_rate=config.event_rate,
                    height=config.height,
                    event2offset={
                        "onset": config.onset_date_offset,
                        "wakeup": config.wakeup_date_offset,
                    },
                    use_daily_norm=config.use_daily_norm,
                    daily_score_offset=config.daily_score_offset,
                    later_date_max_sub_rate=config.later_date_max_sub_rate,
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
                    raise Exception(
                        f"Training failed. epoch: {epoch + 1}, score: {valid_score:.5f}"
                    )

                if debug:
                    break

            logger.info(f"* Best epoch: {best_epoch + 1} \t valid score: {best_score:.5f}")
            cv_scores.append(best_score)
            valid_preds.append(best_valid_fold_preds_df)
            valid_submissions.append(best_valid_fold_submission)

            # save
            best_valid_fold_preds_df.write_parquet(
                data_checkpoint_dir / f"valid_fold{fold}_preds.parquet"
            )
            best_valid_fold_submission.write_parquet(
                data_checkpoint_dir / f"valid_fold{fold}_submission.parquet"
            )

            valid_fold_preds_df.write_parquet(
                data_checkpoint_dir / f"last_valid_fold{fold}_preds.parquet"
            )
            valid_fold_submission.write_parquet(
                data_checkpoint_dir / f"last_valid_fold{fold}_submission.parquet"
            )

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

    # oof score
    oof_valid_preds_df = pl.concat(valid_preds)
    valid_preds_df = oof_valid_preds_df.join(train_df, how="left", on="row_id")
    valid_submission = post_process_from_2nd(
        valid_preds_df,
        event_rate=config.event_rate,
        height=config.height,
        event2offset={"onset": config.onset_date_offset, "wakeup": config.wakeup_date_offset},
        use_daily_norm=config.use_daily_norm,
        daily_score_offset=config.daily_score_offset,
        later_date_max_sub_rate=config.later_date_max_sub_rate,
    )
    valid_score = score(train_events_df, valid_submission)
    logger.info(f"oof score: {valid_score:.5f}")

    # save
    oof_valid_preds_df.sort("row_id").write_parquet(meta_config.save_dir / "valid_preds.parquet")
    pl.concat(valid_submissions).with_columns(pl.arange(0, pl.count()).alias("row_id")).sort(
        "row_id"
    ).write_parquet(meta_config.save_dir / "valid_submission.parquet")

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"all processes done in {elapsed_time:.1f} min.")


def run() -> None:
    # debug
    logger.opt(colors=True).info(
        "<yellow>********************** mode : debug **********************</yellow>"
    )
    main(debug=True)

    # main
    logger.opt(colors=True).info("<yellow>" + "-" * 60 + "</yellow>")
    logger.opt(colors=True).info(
        "<yellow>********************** mode : main **********************</yellow>"
    )
    set_seed(config.seed)
    main(debug=False)


if __name__ == "__main__":
    run()
