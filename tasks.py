from __future__ import annotations

from invoke import Config, task


@task
def prepare_data(c: Config) -> None:
    # input/folds
    # input/periodicity
    pass


@task
def run_kami_1st(c: Config) -> None:
    pass


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
def run_all(c: Config) -> None:
    prepare_data(c)
    run_kami_1st(c)
    run_sakami_1st(c)
    run_shimacos_2nd(c)
    run_sakami_2nd(c)
