import polars as pl


def load_sakami_data(exp_names: list[str]) -> pl.LazyFrame:
    dfs = []
    for exp_name in exp_names:
        df = pl.read_parquet(f"../sakami/output/{exp_name}/valid_preds.parquet").sort("row_id").drop("row_id")
        if "sleep" in df.columns:
            df = df.rename(
                {
                    "prediction_wakeup": f"pred_wakeup_{exp_name}",
                    "prediction_onset": f"pred_onset_{exp_name}",
                    "sleep": f"pred_sleep_{exp_name}",
                }
            )
        else:
            df = df.rename(
                {
                    "prediction_wakeup": f"pred_wakeup_{exp_name}",
                    "prediction_onset": f"pred_onset_{exp_name}",
                }
            )
        dfs.append(df)
    df = pl.concat(dfs, how="horizontal")
    return df


def load_kami_data(exp_names: list[str]) -> pl.LazyFrame:
    df = pl.concat(
        [
            pl.read_parquet(f"../kami/output/cv_inference/{exp_name}/single/train_pred.parquet").rename(
                {
                    "pred_sleep": f"pred_sleep_{exp_name}",
                    "pred_wakeup": f"pred_wakeup_{exp_name}",
                    "pred_onset": f"pred_onset_{exp_name}",
                }
            )
            for exp_name in exp_names
        ],
        how="horizontal",
    )
    return df


def load_shimacos_data(exp_names: list[str]) -> pl.LazyFrame:
    df = pl.concat(
        [
            pl.concat([pl.read_parquet(f"./output/{exp_name}/fold{i}/result/valid.parquet") for i in range(5)])
            .rename(
                {
                    "label_onset_pred": f"pred_onset_{exp_name}",
                    "label_wakeup_pred": f"pred_wakeup_{exp_name}",
                }
            )
            .drop(["label_onset", "label_wakeup"])
            for exp_name in exp_names
        ],
        how="align",
    )
    return df
