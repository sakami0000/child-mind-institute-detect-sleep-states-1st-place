import numpy as np
import polars as pl
from scipy.signal import find_peaks
import logging
from pathlib import Path
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


def post_process_for_seg_group_by_day(keys: list[str], preds: np.ndarray, val_df: pl.DataFrame) -> pl.DataFrame:
    """
    Args:
        keys (list[str]): 予測したchunkのkey({series_id}_{chunk_id})
        preds (np.ndarray): (chunk_num, duration, 2)
        val_df (pl.DataFrame): sequence
    """
    # valid の各ステップに予測結果を付与
    count_df = val_df.get_column("series_id").value_counts()
    series2numsteps_dict = dict(count_df.select("series_id", "counts").iter_rows())

    # 順序を保ったままseries_idを取得
    unique_series_ids = val_df.get_column("series_id").unique(maintain_order=True).to_list()
    key_series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))

    # val_dfに合わせた順番でpredsから予測結果を取得
    preds_list = []
    for series_id in unique_series_ids:
        series_idx = np.where(key_series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)
        this_series_preds = this_series_preds[: series2numsteps_dict[series_id], :]
        preds_list.append(this_series_preds)

    preds_all = np.concatenate(preds_list, axis=0)
    valid_preds_df = val_df.with_columns(
        pl.Series(name="prediction_onset", values=preds_all[:, 0]),
        pl.Series(name="prediction_wakeup", values=preds_all[:, 1]),
    )
    valid_preds_df = valid_preds_df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"))

    # from sakami-san code
    def make_submission(preds_df: pl.DataFrame) -> pl.DataFrame:
        event_dfs = [
            preds_df.with_columns(pl.lit(event).alias("event"), pl.col("timestamp").dt.date().alias("date"))
            .group_by(["series_id", "date"])
            .agg(pl.all().sort_by(f"prediction_{event}").last())
            .rename({f"prediction_{event}": "score"})
            .select(["series_id", "step", "event", "score"])
            for event in ["onset", "wakeup"]
        ]
        submission_df = (
            pl.concat(event_dfs)
            .sort(["series_id", "step"])
            .with_columns(pl.arange(0, pl.count()).alias("row_id"))
            .select(["row_id", "series_id", "step", "event", "score"])
        )
        return submission_df

    submission_df = make_submission(valid_preds_df)

    return submission_df


def post_process_for_seg(
    keys: list[str],
    preds: np.ndarray,
    score_th: float = 0.01,
    distance: int = 5000,
    periodicity_dict: dict[np.ndarray] | None = None,
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        keys (list[str]): list of keys. key is "{series_id}_{chunk_id}"
        preds (np.ndarray): (num_series * num_chunks, duration, 2)
        score_th (float, optional): threshold for score. Defaults to 0.5.
        distance (int, optional): distance for peaks. Defaults to 5000.
        periodicity_dict (dict[np.ndarray], optional): series_id を key に periodicity の 1d の予測結果を持つ辞書. 値は 0 or 1 の np.ndarray. Defaults to None.

    Returns:
        pl.DataFrame: submission dataframe
    """
    LOGGER.info("is periodicity_dict None? : {}".format(periodicity_dict is None))

    series_ids = np.array(list(map(lambda x: x.split("_")[0], keys)))
    unique_series_ids = np.unique(series_ids)

    records = []
    for series_id in unique_series_ids:
        series_idx = np.where(series_ids == series_id)[0]
        this_series_preds = preds[series_idx].reshape(-1, 2)
        if periodicity_dict is not None:
            this_series_preds = this_series_preds[: len(periodicity_dict[series_id]), :]
            this_series_preds *= 1 - periodicity_dict[series_id][:, None]  # periodicity があるところは0にする

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df


def post_process_find_peaks(
    series2preds: dict[np.ndarray],
    score_th: float = 0.01,
    distance: int = 5000,
    periodicity_dict: dict[np.ndarray] | None = None,
) -> pl.DataFrame:
    """make submission dataframe for segmentation task

    Args:
        series2preds (dict[np.ndarray]): series_id を key に 2d の予測結果を持つ辞書
        score_th (float, optional): threshold for score. Defaults to 0.5.
        distance (int, optional): distance for peaks. Defaults to 5000.
        periodicity_dict (dict[np.ndarray], optional): series_id を key に periodicity の 1d の予測結果を持つ辞書. 値は 0 or 1 の np.ndarray. Defaults to None.

    Returns:
        pl.DataFrame: submission dataframe
    """
    LOGGER.info("is periodicity_dict None? : {}".format(periodicity_dict is None))

    records = []
    for series_id in series2preds.keys():
        this_series_preds = series2preds[series_id][:, [1, 2]]
        if periodicity_dict is not None:
            this_series_preds = this_series_preds[: len(periodicity_dict[series_id]), :]
            this_series_preds *= 1 - periodicity_dict[series_id][:, None]  # periodicity があるところは0にする

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_preds = this_series_preds[:, i]
            steps = find_peaks(this_event_preds, height=score_th, distance=distance)[0]
            scores = this_event_preds[steps]
            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:  # 一つも予測がない場合はdummyを入れる
        records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df


def post_process_event_score_normalize_by_day(
    event_df: pl.DataFrame,
    series_df: pl.DataFrame,
    day_start_hour_dict: dict[str, int] = {"onset": 12, "wakeup": 20},
) -> pl.DataFrame:
    """event score を日毎に正規化する。height以上のスコアを持つイベント農地日毎のscoreの合計を1にする

    Args:
        event_df (pl.DataFrame): event score を持つ dataframe
        day_start_hour (dict[str, int], optional): 日付の切り替え時間
        height (float, optional): event score の高さの閾値

    Returns:
        pl.DataFrame: 正規化された event score を持つ dataframe
    """

    # event_df の step カラムの型を u32 に
    event_df = event_df.with_columns(pl.col("step").cast(pl.UInt32))

    # event_df, series_df の series_id, step  を key に、event_df に series_id の timestamp カラムを結合
    event_df = event_df.join(series_df.select(["series_id", "step", "timestamp"]), on=["series_id", "step"])

    result_event_df_list = []

    for event, day_start_hour in day_start_hour_dict.items():
        # event が一致する行を抽出
        one_event_df = event_df.filter(pl.col("event") == event)
        # 日付ごとに day_start_hour だけ時間を引いて、日付カラムを追加
        one_event_df = one_event_df.with_columns(
            pl.col("timestamp").dt.offset_by(f"-{day_start_hour}h").alias("shifted_timestamp")
        ).with_columns(pl.col("shifted_timestamp").dt.date().alias("date"))

        #  score の合計をスコアの正規化に用いる
        score_sum_df = (
            one_event_df.group_by(["series_id", "date"])
            .agg(pl.sum("score").alias("score_sum"))
            .select(["series_id", "date", "score_sum"])
        )
        # 日付ごとの score の合計を event_df に結合
        one_event_df = one_event_df.join(score_sum_df, on=["series_id", "date"])
        # score_sum が欠損している場合は 1 にする
        one_event_df = one_event_df.with_columns(
            pl.when(pl.col("score_sum").is_null()).then(1.0).otherwise(pl.col("score_sum")).alias("score_sum")
        )
        # score_sum が 1 未満の場合は 1 にする（大きくはしない）
        one_event_df = one_event_df.with_columns(
            pl.when(pl.col("score_sum") < 1).then(1.0).otherwise(pl.col("score_sum")).alias("score_sum")
        )

        one_event_df = one_event_df.with_columns(
            pl.when(pl.col("score_sum") > 3.0).then(pl.col("score")).otherwise(pl.col("score") * 0.8).alias("score")
        )
        """
        # 日付ごとの score の合計でスコアを割ると割りすぎなので、power(score_sum) で減衰させる
        # 0.7395099476450617
        one_event_df = one_event_df.with_columns(
            pl.col("score").pow(pl.col("score_sum").alias("score"))
        ) 
        """
        """
        # 日付ごとの score の合計でスコアを割る
        one_event_df = one_event_df.with_columns(
            pl.col("score") / pl.col("score_sum").pow(1.0).alias("score")
        )  # 0.5: 0.7361688492242824, 1.0: 0.7177877757639883,  2.0: 0.6715313626079408
        """

        # 日付カラムを削除
        one_event_df = one_event_df.drop(["shifted_timestamp", "date", "score_sum"])

        # event 結合
        result_event_df_list.append(one_event_df)

    # event を結合
    result_event_df = pl.concat(result_event_df_list)

    return result_event_df


def make_submission(
    preds_df: pl.DataFrame,
    periodicity_dict: dict[str, np.ndarray],
    height: float = 0.001,
    distance: int = 100,
    day_norm: bool = False,
    daily_score_offset: float = 1.0,
    pred_prefix: str = "prediction",
    late_date_rate: float | None = None,
) -> pl.DataFrame:
    event_dfs = []

    for series_id, series_df in tqdm(
        preds_df.group_by("series_id"), desc="find peaks", leave=False, total=len(preds_df["series_id"].unique())
    ):
        for event in ["onset", "wakeup"]:
            event_preds = series_df[f"{pred_prefix}_{event}"].to_numpy().copy()
            event_preds *= 1 - periodicity_dict[series_id][: len(event_preds)]
            steps = find_peaks(event_preds, height=height, distance=distance)[0]
            event_dfs.append(
                series_df.filter(pl.col("step").is_in(steps))
                .with_columns(pl.lit(event).alias("event"))
                .rename({f"{pred_prefix}_{event}": "score"})
                .select(["series_id", "step", "timestamp", "event", "score"])
            )

    submission_df = (
        pl.concat(event_dfs).sort(["series_id", "step"]).with_columns(pl.arange(0, pl.count()).alias("row_id"))
    )

    if day_norm:
        submission_df = submission_df.with_columns(
            pl.col("timestamp").dt.offset_by("2h").dt.date().alias("date")
        ).with_columns(
            pl.col("score") / (pl.col("score").sum().over(["series_id", "event", "date"]) + daily_score_offset)
        )

    if late_date_rate is not None:
        submission_df = (
            submission_df.with_columns(pl.col("timestamp").dt.offset_by("2h").dt.date().alias("date"))
            .with_columns(
                pl.col("date").min().over("series_id").alias("min_date"),
                pl.col("date").max().over("series_id").alias("max_date"),
            )
            .with_columns(
                pl.col("score")
                * (
                    1
                    - (
                        (1 - pl.lit(late_date_rate))
                        * (
                            (pl.col("date") - pl.col("min_date")).dt.days()
                            / ((pl.col("max_date") - pl.col("min_date")).dt.days() + 1.0)
                        )
                    )
                )
            )
        )

    return submission_df.select(["row_id", "series_id", "step", "event", "score"])
