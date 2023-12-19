import glob
import multiprocessing as mp
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning
import torch
import torch.distributed as dist
import yaml
from google.cloud import storage
from omegaconf import DictConfig
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from torch.utils.data.dataloader import DataLoader

plt.style.use("seaborn-whitegrid")


from components.factories import (
    get_dataset,
    get_loss,
    get_model,
    get_optimizer,
    get_sampler,
    get_scheduler,
)
from components.fast_metrics import event_detection_ap
from components.stacking.sync_batchnorm import convert_model
from components.utils import post_process_from_2nd


class BaseRunner(pytorch_lightning.LightningModule):
    def __init__(self, hparams: DictConfig):
        super(BaseRunner, self).__init__()
        self.base_config = hparams.base
        self.data_config = hparams.data
        self.model_config = hparams.model
        self.train_config = hparams.train
        self.test_config = hparams.test
        self.store_config = hparams.store
        # load from factories
        if self.data_config.is_train:
            self.train_dataset = get_dataset(config=self.data_config, mode="train")
            self.valid_dataset = get_dataset(config=self.data_config, mode="valid")
            self.event_df = pl.read_csv("./input/train_events.csv")
            self.event_df = self.event_df.filter(
                (pl.col("series_id").is_in(np.unique(self.valid_dataset.ids).tolist()))
                & pl.col("step").is_not_null()
            )

            if self.data_config.n_fold != "all":
                self.num_train_optimization_steps = int(
                    self.train_config.epoch
                    * len(self.train_dataset)
                    / (self.train_config.batch_size)
                    / self.train_config.accumulation_steps
                    / len(self.base_config.gpu_id)
                )
            else:
                self.num_train_optimization_steps = int(
                    50
                    * len(self.train_dataset)
                    / (self.train_config.batch_size)
                    / self.train_config.accumulation_steps
                    / len(self.base_config.gpu_id)
                )
            if hparams.debug:
                print(self.valid_dataset.__getitem__(0))

        else:
            if self.test_config.is_validation:
                self.test_dataset = get_dataset(config=self.data_config, mode="valid")
                self.prefix = "valid"
            else:
                self.test_dataset = get_dataset(config=self.data_config, mode="test")
                self.prefix = "test"
            self.num_train_optimization_steps = 100
            if hparams.debug:
                print(self.test_dataset.__getitem__(0))
        self.model_config.num_feature = len(hparams.feature.numerical_cols)
        self.model_config.num_category = len(hparams.feature.cat_cols)
        self.model = get_model(self.model_config)
        if self.base_config.loss_class == "nn.BCEWithLogitsLoss":
            self.loss = get_loss(
                loss_class=self.base_config.loss_class,
                pos_weight=torch.ones([1]) * self.train_config.loss_weight,
            )
        elif self.base_config.loss_class == "nn.CrossEntropyLoss":
            # B-とL-に重み付け
            self.loss = get_loss(
                loss_class=self.base_config.loss_class,
                weight=torch.Tensor(
                    [1, 1, self.train_config.loss_weight, self.train_config.loss_weight]
                ),
            )

        # path setting
        self.initialize_variables()
        self.save_flg = False
        self.refinement_step = False

        if len(self.base_config.gpu_id) > 1:
            self.model = convert_model(self.model)
        self.cpu_count = mp.cpu_count() // len(self.base_config.gpu_id)

        # column
        self.id_col = self.data_config.id_col
        self.label_col = self.data_config.label_col
        self.prob_col = self.data_config.prob_col
        self.pred_col = self.data_config.pred_col

        self.step_outputs = []

    def configure_optimizers(self):
        if self.base_config.use_transformer_parameter:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias"]
            optimizer_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.001,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_parameters = self.model.parameters()
        optimizer = get_optimizer(
            opt_class=self.base_config.opt_class,
            params=optimizer_parameters,
            lr=self.train_config.learning_rate,
        )
        if self.base_config.scheduler_class == "GradualWarmupScheduler":
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.num_train_optimization_steps, eta_min=1e-6
            )
            scheduler = {
                "scheduler": get_scheduler(
                    scheduler_class=self.base_config.scheduler_class,
                    optimizer=optimizer,
                    multiplier=self.train_config.warmup_multiplier,
                    total_epoch=int(
                        self.num_train_optimization_steps * self.train_config.warmup_rate
                    ),
                    after_scheduler=scheduler_cosine,
                ),
                "interval": "step",
            }
            scheduler["scheduler"].step(self.step)
        elif self.base_config.scheduler_class == "ReduceLROnPlateau":
            scheduler = {
                "scheduler": get_scheduler(
                    scheduler_class=self.base_config.scheduler_class,
                    optimizer=optimizer,
                    mode=self.train_config.callbacks.mode,
                    factor=0.5,
                    patience=self.train_config.scheduler.patience,
                    verbose=True,
                ),
                "interval": "epoch",
                "monitor": self.train_config.callbacks.monitor_metric,
            }
        else:
            raise NotImplementedError
        return [optimizer], [scheduler]

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.model(batch)

    def train_dataloader(self):
        if self.trainer.num_devices > 1:
            if self.data_config.dataset_class == "query_dataset":
                sampler = get_sampler("weighted_sampler", dataset=self.train_dataset)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                )
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=self.cpu_count,
            sampler=sampler,
            # collate_fn=text_collate,
        )
        return train_loader

    def val_dataloader(self):
        if self.trainer.num_devices > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(
                self.valid_dataset,
            )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.test_config.batch_size,
            num_workers=0,
            pin_memory=self.cpu_count,
            sampler=sampler,
            # collate_fn=text_collate,
        )

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.cpu_count,
            # collate_fn=text_collate,
        )
        return test_loader

    def initialize_variables(self):
        self.step = 0
        if self.train_config.callbacks.mode == "max":
            self.best_score = -np.inf
            self.moninor_op = np.greater_equal
        elif self.train_config.callbacks.mode == "min":
            self.best_score = np.inf
            self.moninor_op = np.less_equal
        if self.train_config.warm_start:
            pass

    def upload_directory(self, path):
        storage_client = storage.Client(self.store_config.gcs_project)
        bucket = storage_client.get_bucket(self.store_config.bucket_name)
        glob._ishidden = lambda x: False
        filenames = glob.glob(f"{path}/**", recursive=True)
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            destination_blob_name = f"{self.store_config.gcs_path}/{filename.split(self.store_config.save_path)[-1][1:]}"
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(filename)

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = (
            running_train_loss.cpu().item() if running_train_loss is not None else float("NaN")
        )
        tqdm_dict = {"loss": "{:2.6g}".format(avg_training_loss)}

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            version = version[-4:] if isinstance(version, str) else version
            tqdm_dict["v_num"] = version

        return tqdm_dict


class ClassificationRunner(BaseRunner):
    def __init__(self, hparams: DictConfig):
        super(ClassificationRunner, self).__init__(hparams)

    def training_step(self, batch, batch_nb):
        pred = self.forward(batch)
        label = batch[self.label_col]
        ignore_mask = label != -1
        loss = self.loss(pred[ignore_mask], label[ignore_mask].float())
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "lr",
            self.optimizers().optimizer.param_groups[0]["lr"],
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_nb):
        if self.test_config.is_tta:
            batch["image"] = torch.cat(
                [batch["image"], batch["image"].flip(1), batch["image"].flip(2)], dim=0
            )
        pred = self.forward(batch)
        if self.test_config.is_tta:
            pred = pred.view((3, -1) + pred.shape[1:]).mean(0)
        label = batch[self.label_col]
        ignore_mask = label != -1
        loss = self.loss(pred[ignore_mask], label[ignore_mask].float())

        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        step = batch["step"].detach().cpu().numpy()
        metrics = {
            self.id_col: batch[self.id_col],
            self.pred_col: pred,
            self.label_col: label,
            "loss": loss,
            "step": step,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature()
            metrics.update({"feature": feature.detach().cpu().numpy()})
        self.step_outputs.append(metrics)

    def test_step(self, batch, batch_nb):
        if self.test_config.is_tta:
            batch["image"] = torch.cat(
                [batch["image"], batch["image"].flip(1), batch["image"].flip(2)], dim=0
            )
        with torch.no_grad():
            pred = self.forward(batch)
        if self.test_config.is_tta:
            pred = pred.view((3, -1) + pred.shape[1:]).mean(0)
        pred = pred.detach().cpu().numpy()
        metrics = {
            self.id_col: batch[self.id_col],
            self.pred_col: pred,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature().detach().cpu().numpy()
            metrics.update({"feature": feature})
        if self.test_config.is_validation:
            label = batch[self.label_col].detach().cpu().numpy()
            metrics.update({self.label_col: label})
        self.step_outputs.append(metrics)

    def _compose_result(self, outputs: List[Dict[str, np.ndarray]]) -> pl.DataFrame:
        def sigmoid(a):
            return 1 / (1 + np.exp(-a))

        steps = np.concatenate([x["step"] for x in outputs], axis=0)
        preds = np.concatenate([x[self.pred_col] for x in outputs], axis=0)
        ids = np.concatenate([x[self.id_col] for x in outputs]).reshape(
            -1,
        )
        df_dict = {self.id_col: ids, "step": steps}
        if self.data_config.use_biol_label:
            for pred_idx in range(self.model_config.num_label):
                df_dict[f"{self.prob_col}_{pred_idx}"] = preds[..., pred_idx]
        elif self.data_config.use_multi_label:
            for pred_idx, col in enumerate(self.data_config.pred_cols):
                df_dict[col] = preds[..., pred_idx]
        else:
            df_dict[self.prob_col] = preds
        if self.label_col in outputs[0].keys():
            if not self.data_config.use_multi_label:
                label = np.concatenate([x[self.label_col] for x in outputs])
                df_dict[self.label_col] = label
            else:
                label = np.concatenate([x[self.label_col] for x in outputs])
                for label_idx, col in enumerate(self.data_config.label_cols):
                    df_dict[col] = label[..., label_idx]

        df = pl.DataFrame(df_dict)
        cols = [col for col in df.columns if col != self.id_col]
        df = df.explode(cols)
        if self.data_config.use_biol_label:
            df = df.group_by([self.id_col, "step"]).agg(
                pl.col(
                    [self.label_col]
                    + [f"{self.prob_col}_{i}" for i in range(self.model_config.num_label)]
                ).mean()
            )
            df = df.with_columns(
                pl.concat_list(f"{self.prob_col}_{i}" for i in range(self.model_config.num_label))
                .list.arg_max()
                .alias(self.pred_col)
            )
        elif self.data_config.use_multi_label:
            df = df.group_by([self.id_col, "step"]).agg(
                pl.col(self.data_config.label_cols + self.data_config.pred_cols).mean()
            )
            df = df.with_columns(
                pl.col([col for col in self.data_config.pred_cols]).map_elements(sigmoid)
            )
        else:
            df = df.group_by([self.id_col, "step"]).agg(
                pl.col([self.prob_col, self.label_col]).mean()
            )
            df = df.with_columns((pl.col(self.prob_col) >= 0).cast(int).alias(self.pred_col))
        df = df.sort([self.id_col, "step"])
        if self.label_col in outputs[0].keys():
            if self.data_config.use_multi_label:
                df = df.filter(pl.col(self.data_config.label_cols[0]) != -1)
            else:
                df = df.filter(pl.col(self.label_col) != -1)
        return df

    def _make_submission(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.data_config.use_biol_label:
            df = df.with_columns(
                (
                    pl.col("label_prob_2").exp()
                    / pl.sum_horizontal(pl.col([f"label_prob_{i}" for i in range(4)]).exp())
                ).alias("onset_prob"),
                (
                    pl.col("label_prob_3").exp()
                    / pl.sum_horizontal(pl.col([f"label_prob_{i}" for i in range(4)]).exp())
                ).alias("wakeup_prob"),
            )
            df = df.with_columns(pl.Series("row_idx", range(len(df))))
            row_idxs = df.group_by(["series_id"]).agg(pl.col(["row_idx"])).rows(named=True)
            preds = df[["onset_prob", "wakeup_prob"]].to_numpy()
            records = []
            zero_series_ids = []
            for row in row_idxs:
                _preds = preds[row["row_idx"]]
                for i, event_name in enumerate(["onset", "wakeup"]):
                    steps, peak_heights = find_peaks(_preds[:, i], height=0.9, distance=5000)
                    if len(steps) > 0:
                        records.append(
                            pl.DataFrame(
                                {
                                    "series_id": row["series_id"],
                                    "event": event_name,
                                    "step": steps,
                                    "score": peak_heights["peak_heights"],
                                }
                            )
                        )
                    else:
                        if row["series_id"] not in zero_series_ids:
                            records.append(
                                pl.DataFrame(
                                    {
                                        "series_id": row["series_id"],
                                        "event": event_name,
                                        "step": 0,
                                        "score": 0.0,
                                    }
                                )
                            )
                            zero_series_ids.append(row["series_id"])
            sub_df = pl.concat(records)
        if self.data_config.use_multi_label:
            df = df.with_columns(pl.Series("row_idx", range(len(df))))
            if "periodicity" in self.data_config.feature.cat_cols:
                periodicity_idx = self.data_config.feature.cat_cols.index("periodicity")
                periodicity_df = pl.DataFrame(
                    {
                        "series_id": self.valid_dataset.ids.tolist(),
                        "step": self.valid_dataset.steps.tolist(),
                        "periodicity": self.valid_dataset.cat_features[:, periodicity_idx].tolist(),
                    }
                )
                df = df.with_columns(pl.col("step").cast(pl.Int64))
                df = df.join(periodicity_df, on=["series_id", "step"], how="left")
                df = df.with_columns(
                    pl.when(pl.col("periodicity") == 1)
                    .then(pl.col("label_onset_pred"))
                    .otherwise(0),
                    pl.when(pl.col("periodicity") == 1)
                    .then(pl.col("label_wakeup_pred"))
                    .otherwise(0),
                )
            if self.data_config.is_stacking:
                df = df.with_columns(
                    ((pl.col("step") - pl.col("step").shift(1)) != 12)
                    .cast(int)
                    .cumsum()
                    .over("series_id")
                    .fill_null(0)
                    .alias("chunk_id")
                )
                row_idxs = (
                    df.group_by(["series_id", "chunk_id"]).agg(pl.col(["row_idx"])).rows(named=True)
                )
            else:
                row_idxs = df.group_by(["series_id"]).agg(pl.col(["row_idx"])).rows(named=True)

            preds = df[["label_onset_pred", "label_wakeup_pred"]].to_numpy()
            steps = df["step"].to_numpy()
            records = []
            zero_series_ids = []
            for row in row_idxs:
                _preds = preds[row["row_idx"]]
                _steps = steps[row["row_id"]]
                for i, event_name in enumerate(["onset", "wakeup"]):
                    idxs, peak_heights = find_peaks(
                        _preds[:, i],
                        height=self.base_config.height,
                        distance=self.base_config.distance,
                    )
                    peak_steps = _steps[idxs]
                    if len(steps) > 0:
                        records.append(
                            pl.DataFrame(
                                {
                                    "series_id": row["series_id"],
                                    "event": event_name,
                                    "step": peak_steps,
                                    "score": peak_heights["peak_heights"],
                                }
                            )
                        )
                    else:
                        if row["series_id"] not in zero_series_ids:
                            records.append(
                                pl.DataFrame(
                                    {
                                        "series_id": row["series_id"],
                                        "event": event_name,
                                        "step": 0,
                                        "score": 0.0,
                                    }
                                )
                            )
                            zero_series_ids.append(row["series_id"])
            sub_df = pl.concat(records)

        else:
            df = df.with_columns(
                pl.col("label_pred").shift(1).over(self.id_col).alias("lag_label_pred")
            )
            df = df.with_columns(
                pl.when((pl.col("label_pred") == 1) & (pl.col("lag_label_pred") == 0))
                .then("onset")
                .when((pl.col("label_pred") == 0) & (pl.col("lag_label_pred") == 1))
                .then("wakeup")
                .alias("event")
            )
            sub_df = df.filter(pl.col("event").is_not_null()).rename({"label_prob": "score"})
            sub_df = sub_df.with_columns(
                pl.when(pl.col("event") == "onset")
                .then(pl.col("score"))
                .otherwise(1 - pl.col("score"))
                .alias("score")
            )
        return sub_df

    def on_validation_epoch_end(self):
        def gaussian_kernel(length, sigma):
            x = np.arange(-length, length + 1)
            h = np.exp(-(x**2) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h

        loss = np.mean([x["loss"].item() for x in self.step_outputs])
        df = self._compose_result(self.step_outputs)
        if self.trainer.num_devices > 1:
            # DDP使用時の結果をまとめる処理
            rank = dist.get_rank()
            df.to_parquet(f"{self.store_config.result_path}/valid_{rank}.parquet")
            dist.barrier()
            metrics = {"avg_loss": loss}
            world_size = dist.get_world_size()
            aggregated_metrics = {}
            for metric_name, metric_val in metrics.items():
                metric_tensor = torch.tensor(metric_val).to(f"cuda:{rank}")
                dist.barrier()
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                reduced_metric = metric_tensor.item() / world_size
                aggregated_metrics[metric_name] = reduced_metric
            loss = aggregated_metrics["avg_loss"]
        else:
            pass
        if self.trainer.num_devices > 1:
            # 各rankでlocalに保存した結果のcsvをまとめる
            paths = sorted(glob.glob(f"{self.store_config.result_path}/valid_[0-9].parquet"))
            df = pl.concat([pl.read_parquet(path) for path in paths])

        metrics = {}
        if self.model_config.num_label == 1:
            f1 = f1_score(df[self.label_col], df[self.pred_col], average="macro")
            metrics["f1_macro"] = float(f1)
            pr_auc = average_precision_score(
                df[self.label_col], df[self.prob_col].to_numpy().reshape(-1, 1)
            )
            metrics["pr_auc"] = float(pr_auc)
        metrics["val_loss"] = float(loss)
        if self.data_config.is_stacking:
            df = df.with_columns(pl.col("step").cast(pl.Int64))
            df = df.join(
                self.valid_dataset.series_step_timestamps, on=["series_id", "step"], how="left"
            )
            df = df.with_columns(
                ((pl.col("step") - pl.col("step").shift(1)) != 12)
                .cast(int)
                .cumsum()
                .over("series_id")
                .fill_null(0)
                .alias("chunk_id")
            )
            sub_df = post_process_from_2nd(
                df, event2col={"onset": "label_onset_pred", "wakeup": "label_wakeup_pred"}
            )
        else:
            sub_df = self._make_submission(df)
        if len(sub_df) > 0:
            metrics["score"] = event_detection_ap(self.event_df.to_pandas(), sub_df.to_pandas())
        else:
            metrics["score"] = 0

        # scoreが改善した時のみ結果を保存
        # df.shape[0] >= 2000はsanity_stepの時はskipさせるため
        if self.moninor_op(metrics[self.train_config.callbacks.monitor_metric], self.best_score):
            self.best_score = metrics[self.train_config.callbacks.monitor_metric]
            self.save_flg = True
            res = {}
            res["step"] = int(self.global_step)
            res["epoch"] = int(self.current_epoch)
            res["best_score"] = self.best_score
            df.write_parquet(f"{self.store_config.result_path}/valid.parquet")
            with open(f"{self.store_config.log_path}/best_score.yaml", "w") as f:
                yaml.dump(res, f, default_flow_style=False)
        for key, val in metrics.items():
            self.log(key, val, prog_bar=True)
        self.step_outputs.clear()
        self.log("best_score", self.best_score, prog_bar=True)

        if self.save_flg:
            if self.store_config.gcs_project is not None:
                pass
            if self.trainer.logger is not None:
                pass
            self.save_flg = False
        if self.train_config.refine_target and not self.train_config.use_gaussian_target:
            self.train_dataset.labels = np.where(
                self.train_dataset.labels == 1.0,
                1.0,
                (self.train_dataset.labels - 1 / self.train_config.epoch).clip(min=0.0),
            )
        elif self.train_config.refine_target and self.train_config.use_gaussian_target:
            labels = np.where(self.train_dataset.labels == 1.0, 1.0, 0)
            for i in range(labels.shape[1]):
                step = 180 * (1 - self.current_epoch / self.train_config.epoch)
                sigma = step / 3.29
                labels[:, i] = np.convolve(labels[:, i], gaussian_kernel(step, sigma), mode="same")
            self.train_dataset.labels = labels
        if self.current_epoch >= self.train_config.refinement_step:
            self.train_dataset.set_refinement_step()

    def on_test_epoch_end(self, outputs):
        df = self._compose_result(outputs)
        if self.store_config.save_feature:
            np.concatenate([x["feature"] for x in outputs], axis=0)
        if self.trainer.num_devices > 1:
            rank = dist.get_rank()
            df.to_parquet(
                f"{self.store_config.result_path}/{self.prefix}_{rank}.parquet", index=False
            )
            dist.barrier()
            paths = sorted(
                glob.glob(f"{self.store_config.result_path}/{self.prefix}_[0-9].parquet")
            )
            df = pl.concat([pl.read_parquet(path) for path in paths])
        df.to_csv(f"{self.store_config.result_path}/{self.prefix}.csv", index=False)
        if not self.test_config.is_validation:
            # testデータに対する推論
            if self.data_config.n_fold == 4:
                pd.read_csv("./input/sample_submission.csv")
                self.upload_directory(self.store_config.root_path)
        else:
            # validationデータに対する推論
            if self.data_config.n_fold == 4:
                dfs = pd.concat(
                    [
                        pd.read_csv(
                            f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/valid.csv"
                        )
                        for i in range(5)
                    ],
                    axis=0,
                )
                dfs.to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}/{self.store_config.model_name}.csv",
                )
                self.upload_directory(self.store_config.root_path)
                return {}

    def on_fit_end(self):
        """
        Called at the very end of fit.
        If on DDP it is called on every process
        """
        if self.store_config.gcs_project is not None:
            self.upload_directory(self.store_config.save_path)
        if self.data_config.n_fold == 4 and self.data_config.is_train:
            dfs = pl.concat(
                [
                    pl.read_parquet(
                        f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/valid.parquet"
                    )
                    for i in range(5)
                ],
            )
            dfs.write_parquet(
                f"{self.store_config.root_path}/{self.store_config.model_name}/valid.parquet",
            )
            if self.data_config.is_stacking:
                sub_df = post_process_from_2nd(
                    dfs, event2col={"onset": "label_onset_pred", "wakeup": "label_wakeup_pred"}
                )
                event_df = pl.read_csv("./input/train_events.csv")
                event_df = event_df.filter(pl.col("step").is_not_null())
                print(event_detection_ap(event_df.to_pandas(), sub_df.to_pandas()))
            if self.store_config.gcs_project is not None:
                self.upload_directory(self.store_config.save_path)
