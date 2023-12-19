from __future__ import annotations

from invoke import Config, task


@task
def prepare_debug_data(c: Config) -> None:
    import polars as pl
    import yaml

    df = pl.read_parquet("./input/train_series.parquet")
    event_df = pl.read_csv("./input/train_events.csv")
    ids = []
    for fold in range(5):
        with open(f"./input/folds/stratify_fold_{fold}.yaml", "r") as f:
            stratify_fold = yaml.safe_load(f)
            ids.append(stratify_fold["train_series_ids"][0])
            ids.append(stratify_fold["valid_series_ids"][0])
            stratify_fold["train_series_ids"] = stratify_fold["train_series_ids"][:1]
            stratify_fold["valid_series_ids"] = stratify_fold["valid_series_ids"][:1]
        with open(f"./input/folds/stratify_fold_{fold}.yaml", "w") as f:
            yaml.safe_dump(stratify_fold, f)
    df = df.filter(pl.col("series_id").is_in(ids))
    event_df = event_df.filter(pl.col("series_id").is_in(ids))
    df.write_parquet("./input/train_series.parquet")
    event_df.write_csv("./input/train_events.csv")


@task
def prepare_data(c: Config, overwrite_folds=False) -> None:
    """
    Preparing the necessary data.
    if overwrite_folds is True, this task overwrites input/folds & kami/run/conf/split. Not reproducible, so results change every time it is run.

    requires:
    - input/child-mind-institute-detect-sleep-states

    outputs:
    - kami/processed/train
    - kami/run/conf/split/stratify_fold_{i}.yaml (optional)
    - input/folds (optional)
    """

    with c.cd("kami"):
        c.run("python -m run.prepare_data phase=train")  # kami/processed/train
        if overwrite_folds:
            c.run("python -m run.split_folds")


@task
def run_kami_1st(c: Config, debug=False) -> None:
    """
    requires:
    - input/child-mind-institute-detect-sleep-states

    outputs:
    - models:
        - kami/output/cv_train/exp068_transformer/cv/bestmodel_fold{fold}.pth
        - kami/output/cv_train/exp078_lstm/cv/bestmodel_fold{fold}.pth
        - kami/output/cv_train/exp081_mixup_short_feat14/cv/bestmodel_fold{fold}.pth
    - oofs:
        - kami/output/cv_inference/exp068_transformer/single/train_pred.parquet
        - kami/output/cv_inference/exp078_lstm/single/train_pred.parquet
        - kami/output/cv_inference/exp081_mixup_short_feat14/single/train_pred.parquet
    """

    with c.cd("kami"):
        num_tta = 1 if debug else 3
        c.run(
            f"python -m run.cv_train exp_name=exp068_transformer 'pos_weight=[1.0, 5.0, 5.0]' batch_size=4 'features=012' model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.3 debug={debug}"
        )
        c.run(
            f"python -m run.cv_inference exp_name=exp068_transformer model.encoder_weights=null model=Spec2DCNNSplit model.n_split=1  duration=17280  batch_size=8 'features=012' num_tta={num_tta} decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 phase=train"
        )
        c.run(
            f"python -m run.cv_train exp_name=exp078_lstm 'pos_weight=[1.0, 5.0, 5.0]' batch_size=8 'features=012' model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor debug={debug}"
        )
        c.run(
            f"python -m run.cv_inference exp_name=exp078_lstm model.encoder_weights=null model=Spec2DCNNSplit model.n_split=1  datamodule.zero_periodicity=True duration=17280  batch_size=8 'features=012' feature_extractor=LSTMFeatureExtractor num_tta={num_tta} phase=train"
        )
        c.run(
            f"python -m run.cv_train exp_name=exp081_mixup_short_feat14 'pos_weight=[1.0, 5.0, 5.0]' batch_size=8 'features=014'  model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 downsample_rate=6 feature_extractor=LSTMFeatureExtractor augmentation.mixup_prob=0.5 'augmentation.mixup_alpha=0.5' offset=5 debug={debug}"
        )
        c.run(
            f"python -m run.cv_inference exp_name=exp081_mixup_short_feat14 model.encoder_weights=null model=Spec2DCNNSplit model.n_split=1  datamodule.zero_periodicity=True duration=17280  batch_size=8 'features=014' feature_extractor=LSTMFeatureExtractor downsample_rate=6 num_tta={num_tta} phase=train"
        )


@task
def run_sakami_1st(c: Config) -> None:
    """
    requires:
    - input/folds
    - input/periodicity

    outputs:
    - output/148_gru_scale_factor/valid_preds.parquet
    - output/156_gru_transformer_residual/valid_preds.parquet
    - output/163_gru_sleep_target/valid_preds.parquet
    - output/179_gru_minute_embedding_sync/valid_preds.parquet
    """

    with c.cd("sakami"):
        c.run("python -m exp.148_gru_scale_factor")
        c.run("python -m exp.156_gru_transformer_residual")
        c.run("python -m exp.163_gru_sleep_target")
        c.run("python -m exp.179_gru_minute_embedding_sync")


@task
def create_stacking_data(c: Config) -> None:
    """
    requires:
    - kami/output/cv_inference/exp068_transformer/single/train_pred.parquet
    - kami/output/cv_inference/exp078_lstm/single/train_pred.parquet
    - kami/output/cv_inference/exp081_mixup_short_feat14/single/train_pred.parquet
    - output/148_gru_scale_factor/valid_preds.parquet
    - output/156_gru_transformer_residual/valid_preds.parquet
    - output/163_gru_sleep_target/valid_preds.parquet
    - output/179_gru_minute_embedding_sync/valid_preds.parquet
    outputs:
    - input/preprocess/fold_v2.parquet
    - input/stacking/009.parquet
    - input/stacking/030_truncate.parquet
    """
    with c.cd("shimacos"):
        c.run("python -m components.preprocess.make_fold_v2")
        c.run("python -m components.stacking.009")
        c.run("python -m components.stacking.030_truncate")


@task
def run_shimacos_2nd(c: Config) -> None:
    """
    requires:
    - input/stacking/030_truncate.pkl
    outputs:
    - model
        - shimacos/output/stacking_exp059_030_truncate_lgbm/model/booster_onset.pkl
        - shimacos/output/stacking_exp059_030_truncate_lgbm/model/booster_wakeup.pkl
        - shimacos/output/stacking_exp060_030_truncate_small/fold*/model/*.ckpt
        - shimacos/output/stacking_exp061_030_truncate_cat/model/booster_label_onset.pkl
        - shimacos/output/stacking_exp061_030_truncate_cat/model/booster_label_wakeup.pkl
    - oof
        - shimacos/output/stacking_exp059_030_truncate_lgbm/result/pred_onset.parquet
        - shimacos/output/stacking_exp059_030_truncate_lgbm/result/pred_wakeup.parquet
        - shimacos/output/stacking_exp060_030_truncate_small/valid.parquet
        - shimacos/output/stacking_exp061_030_truncate_cat/result/pred_onset.parquet
        - shimacos/output/stacking_exp061_030_truncate_cat/result/pred_wakeup.parquet

    """
    with c.cd("shimacos"):
        c.run("./bin/stacking_exp059_030_truncate_lgbm.sh")
        c.run("./bin/stacking_exp060_030_truncate_small.sh")
        c.run("./bin/stacking_exp061_030_truncate_cat.sh")


@task
def run_sakami_2nd(c: Config) -> None:
    """
    requires:
    - input/folds
    - input/periodicity
    - input/stacking/009.parquet
    """

    with c.cd("sakami"):
        c.run("python -m stacking.004_transformer_category_padding_idx")
        c.run("python -m stacking.011_cnn_embedding_sync")


@task
def run_all(c: Config, overwrite_folds=False) -> None:
    prepare_data(c, overwrite_folds)
    run_kami_1st(c)
    run_sakami_1st(c)
    create_stacking_data(c)
    run_shimacos_2nd(c)
    run_sakami_2nd(c)
