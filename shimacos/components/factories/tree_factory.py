import logging
import os
import pickle
from typing import Dict, Optional, Tuple

import catboost as cat
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from components.fast_metrics import event_detection_ap
from components.utils import post_process_from_2nd

plt.style.use("seaborn-whitegrid")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(name)s] [L%(lineno)d] [%(levelname)s][%(funcName)s] %(message)s "
    )
)
logger.addHandler(handler)


class LGBMModel(object):
    """
    label_col毎にlightgbm modelを作成するためのクラス
    """

    def __init__(
        self,
        config: DictConfig,
        event_df: pl.DataFrame | None = None,
        event_name: str | None = None,
    ):
        self.config = config
        self.model_dicts: Dict[int, lgb.Booster] = {}
        self.feature_cols = config.numerical_cols + config.cat_cols
        self.event_df = event_df
        self.event_name = event_name

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def _custom_objective(self, preds: np.ndarray, data: lgb.Dataset):
        labels = data.get_label()
        weight = data.get_weight()
        grad = 2 * weight * (preds - labels)
        hess = 2 * weight
        return grad, hess

    def _custom_metric(self, y_pred: np.ndarray, dtrain: lgb.basic.Dataset):
        y_true = dtrain.get_label().astype(float)
        y_pred = y_pred.reshape(self.config.params.num_class, -1).argmax(0)
        score = f1_score(y_true, y_pred, average="micro")
        return "micro_f1", score, True

    def cv(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame | None = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        importances = []
        if self.config.params.num_class == 1:
            oof_preds = np.zeros(len(train_df))
        else:
            oof_preds = np.zeros((len(train_df), self.config.params.num_class))

        if test_df is not None:
            if self.config.params.num_class == 1:
                test_preds = np.zeros(len(test_df))
            else:
                test_preds = np.zeros((len(test_df), self.config.params.num_class))
        train_df = train_df.with_columns(pl.Series("row_id", range(len(train_df))))
        for n_fold in range(self.config.n_fold):
            if self.config.batch_fit:
                bst = self.batch_fit(train_df, n_fold)
            else:
                bst = self.fit(train_df, n_fold)
            valid_df = train_df.filter(pl.col("fold") == n_fold)
            row_idxs = valid_df["row_id"].to_numpy()
            if self.config.params.objective == "multiclass":
                preds = bst.predict(valid_df[self.feature_cols].to_numpy())
                for i in range(self.config.params.num_class):
                    oof_preds[row_idxs, i] = preds[:, i]
                if test_df is not None:
                    _test_preds = bst.predict(test_df[self.feature_cols].to_numpy())
                    for i in range(self.config.params.num_class):
                        test_preds[:, i] += _test_preds[:, i] / self.config.n_fold
            else:
                oof_preds[row_idxs] = bst.predict(valid_df[self.feature_cols].to_numpy())
                if test_df is not None:
                    test_preds += (
                        bst.predict(test_df[self.feature_cols].to_numpy()) / self.config.n_fold
                    )
            self.store_model(bst, n_fold)
            importances.append(bst.feature_importance(importance_type="gain"))
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        train_df = train_df.with_columns(pl.Series(self.config.pred_col, oof_preds))
        if test_df is not None:
            test_df = test_df.with_columns(pl.Series(self.config.pred_col, test_preds))
            return train_df, test_df
        else:
            return train_df

    def fit(
        self,
        df: pl.DataFrame,
        n_fold: int,
    ) -> lgb.Booster:
        params = dict(self.config.params)
        train_df = df.filter(pl.col("fold") != n_fold)
        valid_df = df.filter(pl.col("fold") == n_fold)
        if self.config.negative_sampling:
            pos_df = train_df.filter(pl.col(self.config.label_col) == 1)
            neg_df = train_df.filter(pl.col(self.config.label_col) != 1).sample(fraction=0.1)
            train_df = pl.concat([pos_df, neg_df])
        X_train = train_df[self.feature_cols].to_numpy()
        y_train = train_df[self.config.label_col].to_numpy()

        X_valid = valid_df[self.feature_cols].to_numpy()
        y_valid = valid_df[self.config.label_col].to_numpy()
        logger.info(
            f"{self.config.label_col} [Fold {n_fold}] train shape: {X_train.shape}, valid shape: {X_valid.shape}"
        )
        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            # weight=train_df.query("fold!=@n_fold")["weights"].values,
            feature_name=self.feature_cols,
        )
        lgvalid = lgb.Dataset(
            X_valid,
            label=np.array(y_valid),
            # weight=train_df.query("fold==@n_fold")["weights"].values,
            feature_name=self.feature_cols,
        )
        if self.config.params.objective == "lambdarank":
            params["label_gain"] = list(range(int(df[self.config.label_col].max() + 1)))
            # params["lambdarank_truncation_level"] = int(df[self.config.label_col].max() + 1)
            train_group = (
                train_df.group_by(["series_id", "gt_step"], maintain_order=True)
                .agg(pl.count())["count"]
                .to_list()
            )
            valid_group = (
                valid_df.group_by(["series_id", "gt_step"], maintain_order=True)
                .agg(pl.count())["count"]
                .to_list()
            )
            lgtrain.set_group(train_group)
            lgvalid.set_group(valid_group)
            params["ndcg_eval_at"] = [5, 10]
        bst = lgb.train(
            params,
            lgtrain,
            num_boost_round=100000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            categorical_feature=self.config.cat_cols,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(self.config.verbose_eval),
            ],
        )
        logger.info(
            f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
        )
        return bst

    def batch_fit(
        self,
        df: pl.DataFrame,
        n_fold: int,
    ) -> lgb.Booster:
        params = dict(self.config.params)
        train_df = df.filter(pl.col("fold") != n_fold)
        valid_df = df.filter(pl.col("fold") == n_fold)
        if self.config.negative_sampling:
            pos_df = train_df.filter(pl.col(self.config.label_col) == 1)
            neg_df = train_df.filter(pl.col(self.config.label_col) != 1).sample(fraction=0.1)
            train_df = pl.concat([pos_df, neg_df])
        X_train = train_df[self.feature_cols].to_numpy()
        y_train = train_df[self.config.label_col].to_numpy()

        X_valid = valid_df[self.feature_cols].to_numpy()
        y_valid = valid_df[self.config.label_col].to_numpy()
        logger.info(
            f"{self.config.label_col} [Fold {n_fold}] train shape: {X_train.shape}, valid shape: {X_valid.shape}"
        )
        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            # weight=train_df.query("fold!=@n_fold")["weights"].values,
            feature_name=self.feature_cols,
            free_raw_data=False,
        )
        lgvalid = lgb.Dataset(
            X_valid,
            label=np.array(y_valid),
            # weight=train_df.query("fold==@n_fold")["weights"].values,
            feature_name=self.feature_cols,
            free_raw_data=False,
        )
        if self.config.params.objective == "lambdarank":
            params["label_gain"] = list(range(int(df[self.config.label_col].max() + 1)))
            # params["lambdarank_truncation_level"] = int(df[self.config.label_col].max() + 1)
            train_group = (
                train_df.group_by(["series_id", "gt_step"], maintain_order=True)
                .agg(pl.count())["count"]
                .to_list()
            )
            valid_group = (
                valid_df.group_by(["series_id", "gt_step"], maintain_order=True)
                .agg(pl.count())["count"]
                .to_list()
            )
            lgtrain.set_group(train_group)
            lgvalid.set_group(valid_group)
            params["ndcg_eval_at"] = [5, 10]
        bst = None
        best_score = -np.inf
        tolerance = 0
        best_iteration = 0
        _event_df = self.event_df.filter(
            pl.col("series_id").is_in(valid_df["series_id"].unique())
            & (pl.col("event") == self.event_name)
        ).filter(pl.col("step").is_not_null())
        for i in range(0, 100000, self.config.batch_iter):
            bst = lgb.train(
                params,
                lgtrain,
                num_boost_round=self.config.batch_iter,
                valid_sets=[lgtrain, lgvalid],
                valid_names=["train", "valid"],
                categorical_feature=self.config.cat_cols,
                callbacks=[
                    lgb.log_evaluation(self.config.verbose_eval),
                ],
                init_model=bst,
            )
            valid_df = valid_df.with_columns(
                pl.Series(
                    self.config.pred_col,
                    bst.predict(X_valid, num_iteration=(i + 1) * self.config.batch_iter),
                )
            )
            sub_df = post_process_from_2nd(
                valid_df,
                event2col={self.event_name: self.config.pred_col},
            )
            score = event_detection_ap(
                _event_df.to_pandas(),
                sub_df.to_pandas(),
                tolerances={
                    self.event_name: [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
                },
            )
            logger.info(f"iter: {i}, score: {score}")
            if score > best_score:
                best_score = score
                best_iteration = i + self.config.batch_iter
            else:
                tolerance += 1
                if tolerance >= 2:
                    bst.best_iteration = best_iteration
                    break

        logger.info(f"best_itelation: {best_iteration}, valid: {best_score}")
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(xerr="std", figsize=(10, 20))
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )


class XGBModel(object):
    """
    label_col毎にxgboost modelを作成するようのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, xgb.Booster] = {}

    def custom_metric(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label().astype(float)
        score = average_precision_score(y_true, y_pred)
        return "average_precision_score", -score

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        importances = []
        for n_fold in range(self.config.n_fold):
            bst = self.fit(train_df, n_fold, pseudo_df)
            valid_df = train_df.query("fold == @n_fold")
            if self.config.params.objective == "multi:softmax":
                preds = bst.predict(xgb.DMatrix(valid_df[self.feature_cols]))
                test_preds = bst.predict(xgb.DMatrix(test_df[self.feature_cols]))
                for i in range(self.config.params.num_class):
                    train_df.loc[valid_df.index, f"{self.config.label_col}_prob{i}"] = preds[:, i]
                    test_df[f"{self.config.label_col}_prob{i}_fold{n_fold}"] = test_preds[:, i]
            else:
                train_df.loc[valid_df.index, self.config.pred_col] = bst.predict(
                    xgb.DMatrix(valid_df[self.feature_cols])
                )
                test_df[f"{self.config.pred_col}_fold{n_fold}"] = bst.predict(
                    xgb.DMatrix(test_df[self.feature_cols])
                )
            self.store_model(bst, n_fold)
            importance_dict = bst.get_score(importance_type="gain")
            importances.append(
                [importance_dict[col] if col in importance_dict else 0 for col in self.feature_cols]
            )
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)

        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> xgb.Booster:
        X_train = train_df.query("fold!=@n_fold")[self.feature_cols]
        y_train = train_df.query("fold!=@n_fold")[self.config.label_col]

        X_valid = train_df.query("fold==@n_fold")[self.feature_cols]
        y_valid = train_df.query("fold==@n_fold")[self.config.label_col]
        print("=" * 10, self.config.label_col, n_fold, "=" * 10)
        dtrain = xgb.DMatrix(
            X_train,
            label=np.array(y_train),
            feature_names=self.feature_cols,
        )
        dvalid = xgb.DMatrix(
            X_valid,
            label=np.array(y_valid),
            feature_names=self.feature_cols,
        )
        bst = xgb.train(
            self.config.params,
            dtrain,
            num_boost_round=50000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=50,
            # feval=self.custom_metric,
        )
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(xerr="std", figsize=(10, 20))
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )


class CatModel(object):
    """
    label_col毎にcatboost modelを作成するためのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, cat.Booster] = {}
        self.feature_cols = config.numerical_cols + config.cat_cols

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        df: pd.DataFrame,
        test_df: Optional[pl.DataFrame] = None,
        pseudo_df: Optional[pl.DataFrame] = None,
    ) -> pd.DataFrame:
        importances = []
        preds = np.zeros(len(df))
        df = df.with_columns(pl.arange(0, len(df)).alias("index"))
        for n_fold in range(self.config.n_fold):
            train_df = df.filter(df["fold"] != n_fold)
            valid_df = df.filter(df["fold"] == n_fold)
            logger.info(
                f"{self.config.label_col}[fold {n_fold}] train shape: {train_df.shape}, valid shape: {valid_df.shape}"
            )
            bst, importance = self.fit(train_df, valid_df, pseudo_df)
            valid_pool = cat.Pool(
                valid_df[self.feature_cols].to_pandas(),
                feature_names=self.feature_cols,
                cat_features=list(self.config.categorical_features_indices),
            )
            preds[valid_df["index"].to_numpy()] = bst.predict(
                valid_pool, prediction_type="Probability"
            )[:, 1]
            if test_df is not None:
                test_preds = bst.predict(
                    cat.Pool(
                        test_df[self.feature_cols].to_pandas(),
                        feature_names=self.feature_cols,
                        cat_features=list(self.config.categorical_features_indices),
                    ),
                    prediction_type="Probability",
                )[:, 1]
                test_df = test_df.with_columns(
                    pl.Series(f"{self.config.pred_col}_fold{n_fold}", test_preds)
                )
            self.store_model(bst, n_fold)
            importances.append(importance)
        df = df.with_columns(pl.Series(self.config.pred_col, preds))
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)

        if test_df is not None:
            return df, test_df
        else:
            return df

    def fit(
        self,
        train_df: pl.DataFrame,
        valid_df: pl.DataFrame,
        pseudo_df: Optional[pl.DataFrame] = None,
    ) -> cat.CatBoost:
        if self.config.negative_sampling:
            pos_df = train_df.filter(pl.col(self.config.label_col) == 1)
            neg_df = train_df.filter(pl.col(self.config.label_col) != 1).sample(fraction=0.1)
            train_df = pl.concat([pos_df, neg_df])
        X_train = train_df.select(self.feature_cols)
        y_train = train_df.select(self.config.label_col)

        X_valid = valid_df.select(self.feature_cols)
        y_valid = valid_df.select(self.config.label_col)
        dtrain = cat.Pool(
            data=X_train.to_pandas(),
            label=y_train.to_numpy(),
            feature_names=self.feature_cols,
            cat_features=list(self.config.categorical_features_indices),
        )
        dvalid = cat.Pool(
            X_valid.to_pandas(),
            label=y_valid.to_numpy(),
            feature_names=self.feature_cols,
            cat_features=list(self.config.categorical_features_indices),
        )
        if self.config.params.loss_function in [
            "YetiRank",
            "PairLogit",
            "PairLogitPairwise",
        ]:
            dtrain.set_group_id(train_df["series_id"].to_numpy())
            dvalid.set_group_id(valid_df["series_id"].to_numpy())

        bst = cat.train(
            pool=dtrain,
            params=dict(self.config.params),
            evals=dvalid,
            early_stopping_rounds=100,
            verbose_eval=100,
            # feval=self.custom_metric,
        )
        importance = bst.get_feature_importance(dtrain, type="FeatureImportance")
        return bst, importance

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster_{self.config.label_col + suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(xerr="std", figsize=(10, 20))
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )
