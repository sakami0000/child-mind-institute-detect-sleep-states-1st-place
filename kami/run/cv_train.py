import gc
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from src.datamodule.centernet import CenterNetDataModule
from src.datamodule.seg import SegDataModule
from src.datamodule.seg_cat import SegDataModule as SegDataModuleCat
from src.datamodule.seg_overlap import SegDataModule as SegDataModuleOverlap
from src.datamodule.seg_stride import SegDataModule as SegDataModuleStride
from src.modelmodule.seg import SegModel
from src.modelmodule.seg_cat import SegModel as SegModelCat
from src.utils.metrics import event_detection_ap
from src.utils.periodicity import get_periodicity_dict
from src.utils.post_process import post_process_for_seg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="cv_train", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    seed_everything(cfg.seed)

    # init experiment logger
    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project="child-mind-institute-detect-sleep-states-v2"
        if cfg.num_fold == 5
        else "child-mind-institute-detect-sleep-states-v2-10fold",
        mode="disabled",
    )
    pl_logger.log_hyperparams(cfg)

    for fold in range(cfg.num_fold):
        LOGGER.info(f"Start Training Fold {fold}")
        # init lightning model
        if cfg.model.name == "CenterNet":
            datamodule = CenterNetDataModule(cfg, fold)
        elif cfg.datamodule.how == "cat":
            datamodule = SegDataModuleCat(cfg, fold)
        elif cfg.datamodule.how == "random":
            datamodule = SegDataModule(cfg, fold)
        elif cfg.datamodule.how == "stride":
            datamodule = SegDataModuleStride(cfg, fold)
        elif cfg.datamodule.how == "overlap":
            datamodule = SegDataModuleOverlap(cfg, fold)

        LOGGER.info("Set Up DataModule")

        if cfg.datamodule.how == "cat":
            model = SegModelCat(
                cfg,
                datamodule.valid_event_df,
                len(cfg.features),
                len(cfg.labels),
                cfg.duration,
                datamodule,
                fold,
            )
        else:
            model = SegModel(
                cfg,
                datamodule.valid_event_df,
                len(cfg.features),
                len(cfg.labels),
                cfg.duration,
                datamodule,
                fold,
            )

        # set callbacks
        checkpoint_cb = ModelCheckpoint(
            verbose=True,
            monitor=f"{cfg.monitor}_fold{fold}",
            mode=cfg.monitor_mode,
            save_top_k=1,
            save_last=False,
        )
        lr_monitor = LearningRateMonitor("epoch")
        progress_bar = RichProgressBar()
        model_summary = RichModelSummary(max_depth=2)

        trainer = Trainer(
            # env
            default_root_dir=Path.cwd(),
            # num_nodes=cfg.training.num_gpus,
            accelerator=cfg.accelerator,
            precision=16 if cfg.use_amp else 32,
            # training
            max_epochs=1 if cfg.debug else cfg.epoch,
            max_steps=cfg.epoch * len(datamodule.train_dataloader()),
            gradient_clip_val=cfg.gradient_clip_val,
            accumulate_grad_batches=cfg.accumulate_grad_batches,
            callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
            logger=pl_logger,
            # resume_from_checkpoint=resume_from,
            num_sanity_val_steps=0,
            log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
            reload_dataloaders_every_n_epochs=1,
            sync_batchnorm=True,
            check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        )

        trainer.fit(model, datamodule=datamodule)
        del model
        del trainer
        del datamodule
        gc.collect()

    # oof での評価
    LOGGER.info("Start OOF scoring")

    # 正解ラベルの読み込み
    train_event_df = pl.read_csv(
        Path(cfg.dir.data_dir) / "train_events.csv"
    ).drop_nulls()

    # 予測結果の読み込み
    keys_list = []
    preds_list = []
    for fold in range(cfg.num_fold):
        preds_list.append(np.load(f"preds_fold{fold}.npy"))
        keys_list.append(np.load(f"keys_fold{fold}.npy"))
    preds = np.concatenate(preds_list, axis=0)
    keys = np.concatenate(keys_list, axis=0)

    # 評価
    LOGGER.info("Start event_detection_ap")
    oof_event_df = post_process_for_seg(
        keys,
        preds[:, :, [1, 2]],
        score_th=cfg.post_process.score_th,
        distance=cfg.post_process.distance,
    )
    score, ap_table = event_detection_ap(
        train_event_df.to_pandas(),
        oof_event_df.to_pandas(),
        with_table=True,
    )
    wandb.log(
        {
            "ap_table": wandb.Table(
                dataframe=ap_table.reset_index()[["event", "tolerance", "ap"]]
            )
        }
    )
    LOGGER.info(f"OOF score: {score}")
    wandb.log({"cv_score": score})

    # periodicity
    periodicity_dict = get_periodicity_dict(cfg)
    oof_event_df = post_process_for_seg(
        keys,
        preds[:, :, [1, 2]],
        score_th=cfg.post_process.score_th,
        distance=cfg.post_process.distance,
        periodicity_dict=periodicity_dict,
    )
    score, ap_table = event_detection_ap(
        train_event_df.to_pandas(),
        oof_event_df.to_pandas(),
        with_table=True,
    )
    wandb.log(
        {
            "ap_table_wo_periodicity": wandb.Table(
                dataframe=ap_table.reset_index()[["event", "tolerance", "ap"]]
            )
        }
    )
    LOGGER.info(f"OOF score with out periodicity: {score}")
    wandb.log({"cv_score_wo_periodicity": score})

    for event in ["onset", "wakeup"]:
        plt.figure(figsize=(10, 10))
        for (event_key, tolerance), group in ap_table[
            ap_table.index.get_level_values("event") == event
        ].iterrows():
            plt.plot(
                group["recall"][:-1],
                group["precision"][:-1],
                label=f'Tolerance: {tolerance}, AP:{group["ap"]:.3f}',
            )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curves for {event}")
        plt.legend()
        # 図をwandbにログとして保存
        wandb.log({f"pr_curve_{event}": wandb.Image(plt)})
        plt.close()

    return


if __name__ == "__main__":
    main()
