from __future__ import annotations

from invoke import Config, task


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
        c.run("python run/prepare_data.py phase=train")  # kami/processed/train
        if overwrite_folds:
            c.run("python run/split_folds.py")


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
        - kami/output/cv_inference/exp068_transformer/cv/train_pred.parquet
        - kami/output/cv_inference/exp078_lstm/cv/train_pred.parquet
        - kami/output/cv_inference/exp081_mixup_short_feat14/cv/train_pred.parquet
    """

    with c.cd("kami"):
        num_tta = 1 if debug else 3
        c.run(
            f"python run/cv_train.py exp_name=exp068_transformer 'pos_weight=[1.0, 5.0, 5.0]' batch_size=4 'features=012' model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 decoder.dropout=0.3 debug={debug}"
        )
        c.run(
            f"python -m run.cv_inference exp_name=exp068_transformer model.encoder_weights=null model=Spec2DCNNSplit model.n_split=1  duration=17280  batch_size=8 'features=012' num_tta={num_tta} decoder=TransformerDecoder downsample_rate=2 decoder.num_layers=3 phase=train"
        )
        c.run(
            f"python run/cv_train.py exp_name=exp078_lstm 'pos_weight=[1.0, 5.0, 5.0]' batch_size=8 'features=012' model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 feature_extractor=LSTMFeatureExtractor debug={debug}"
        )
        c.run(
            f"python -m run.cv_inference exp_name=exp078_lstm model.encoder_weights=null model=Spec2DCNNSplit model.n_split=1  datamodule.zero_periodicity=True duration=17280  batch_size=8 'features=012' feature_extractor=LSTMFeatureExtractor num_tta={num_tta} phase=train"
        )
        c.run(
            f"python run/cv_train.py exp_name=exp081_mixup_short_feat14 'pos_weight=[1.0, 5.0, 5.0]' batch_size=8 'features=014'  model=Spec2DCNNSplit model.n_split=1 epoch=30 monitor=val_score monitor_mode=max duration=17280 datamodule.zero_periodicity=True decoder.dropout=0.3 downsample_rate=6 feature_extractor=LSTMFeatureExtractor augmentation.mixup_prob=0.5 'augmentation.mixup_alpha=0.5' offset=5 debug={debug}"
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
def run_shimacos_2nd(c: Config) -> None:
    pass


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
    run_shimacos_2nd(c)
    run_sakami_2nd(c)
