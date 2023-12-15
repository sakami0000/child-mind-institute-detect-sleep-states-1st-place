import numpy as np
import polars as pl
from tqdm.auto import tqdm
from numba import jit


def post_process_from_2nd(
    pred_df,
    event_rate: int | float = 500,
    height: float = 0.001,
    event2col: dict[str, str] = {"onset": "stacking_prediction_onset", "wakeup": "stacking_prediction_wakeup"},
    event2offset: dict[str, str] = {"onset": "5h", "wakeup": "0h"},
    use_daily_norm: bool = True,
    daily_score_offset=10,
    later_date_max_sub_rate: float | None = 0.05,
    tqdm_disable: bool = False,
):
    """
    1分ごとの予測値を用いてイベントを検出する
    用語
    - 予測地点: 2段目のモデルによって得られた1分毎の予測位置
    - 候補地点: event の候補となる 15秒 or 45秒始まりの30秒間隔の位置

    Args:
        pred_df (pl.DataFrame): timestamp 込み
        event_rate (int | float, optional): [0,1) の値であれば1分間に何回イベントが起こるか。intの場合はseries_idごとに同じイベント数を検出。 Defaults to 0.005.
        height (float, optional): 候補地点の期待値がこの値を下回ったら終了。 Defaults to 0.1.
        event2col (dict[str, str], optional): event名と予測値のカラム名の対応。 Defaults to {"onset": "stacking_prediction_onset", "wakeup": "stacking_prediction_wakeup"}.
        weight_rate (float | None, optional): 遠くの予測値の期待値を割り引く際の重み。Noneの場合は重みを1とする。1/weight_rate 倍ずつ遠くの予測値の重みが小さくなっていく。 Defaults to None.
        use_daily_norm (bool, optional): 一日ごとに予測値を正規化するかどうか。 Defaults to False.
        daily_score_offset (float, optional): 正規化の際のoffset。 Defaults to 1.0.
        later_date_max_sub_rate (float | None, optional): 日付が古いほど予測値を割り引く際の最大割引率。Noneの場合は割引しない。 Defaults to None.
    Returns:
        event_df (pl.DataFrame): row_id, series_id, step, event, score をカラムに持つ。
    """
    high_match_nums = (0, 1, 3, 5, 8, 10, 13, 15, 20, 25, 30)
    low_match_nums = (0, 1, 3, 5, 7, 10, 12, 15, 20, 25, 30)
    match_sums = np.ones(10)
    result_events_records = []

    # event ごとに処理
    for event, event_pred_col in event2col.items():
        """
        元の系列の予測地点(長さN): 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, ..., (N-1)*12
        15秒から30秒おきのevent候補地点(長さ2N): 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, ..., (N-1)*12+3, (N-1)*12+9
            - 15秒(3step)から1分おき(長さN): 3, 15, 27, 39, 51, 63, 75, 87, 99, 111, ..., (N-1)*12+3
                - 左の個数 {12: 1, 36: 3, 60: 5, 90: *8*, 120: 10, 150: *13*, 180: 15, 240: 20, 300: 25, 360: 30} high_match_nums
                - 右の個数 {12: 1, 36: 3, 60: 5, 90: *7*, 120: 10, 150: *12*, 180: 15, 240: 20, 300: 25, 360: 30} low_match_nums
            - 45秒(9step)から1分おき(長さN): 9, 21, 33, 45, 57, 69, 81, 93, 105, 117, ..., (N-1)*12+9
                - 左の個数 {12: 1, 36: 3, 60: 5, 90: *7*, 120: 10, 150: *12*, 180: 15, 240: 20, 300: 25, 360: 30} low_match_nums
                - 右の個数 {12: 1, 36: 3, 60: 5, 90: *8*, 120: 10, 150: *13*, 180: 15, 240: 20, 300: 25, 360: 30} high_match_nums       
        """

        # series内でのindexを振り、chunk内での最大と最小を計算
        minute_pred_df = pred_df.filter(pl.col("timestamp").is_not_null())

        if use_daily_norm:
            minute_pred_df = (
                minute_pred_df.with_columns(
                    pl.col("timestamp").dt.offset_by(event2offset[event]).dt.date().alias("date")
                )
                .with_columns(pl.col(event_pred_col).sum().over(["series_id", "date"]).alias("date_sum"))
                .with_columns(
                    pl.col(event_pred_col) / (pl.col("date_sum") + (1 / (daily_score_offset + pl.col("date_sum"))))
                )
            )
        if later_date_max_sub_rate is not None:
            minute_pred_df = minute_pred_df.with_columns(
                pl.col("date").min().over("series_id").alias("min_date"),
                pl.col("date").max().over("series_id").alias("max_date"),
            ).with_columns(
                pl.col(event_pred_col)
                * (
                    pl.lit(1.0)
                    - (
                        pl.lit(later_date_max_sub_rate)
                        * (
                            (pl.col("date") - pl.col("min_date")).dt.days().cast(float)
                            / ((pl.col("max_date") - pl.col("min_date")).dt.days().cast(float) + 1.0)
                        )
                    )
                )
            )

        max_event_per_series = event_rate if isinstance(event_rate, int) else int(len(minute_pred_df) * event_rate)

        # series_id, chunk_id, step でソート
        minute_pred_df = minute_pred_df.sort(["series_id", "chunk_id", "step"])

        # 1. 期待値の計算
        # 1.1 左側を計算 (同じindexの予測を含む左側を計算)
        """
        以下をそれぞれ計算する
        - 15秒(3step)から1分おき(長さN)での候補地点での期待値: stepは 3, 15, 27, 39, 51, 63, 75, 87, 99, 111, ..., (N-1)*12+3
        - 45秒(9step)から1分おき(長さN)での候補地点での期待値: stepは 9, 21, 33, 45, 57, 69, 81, 93, 105, 117, ..., (N-1)*12+9
        計算は左側の予測地点の数と、右側の予測地点の数
        """
        minute_pred_df = minute_pred_df.with_columns(
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over(["series_id", "chunk_id"])
                        / match_sums[i]
                    )
                    for i, window in enumerate(high_match_nums[1:])
                ]
            ).alias(f"{event}_left_expectation_plus_3step"),
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over(["series_id", "chunk_id"])
                        / match_sums[i]
                    )
                    for i, window in enumerate(low_match_nums[1:])
                ]
            ).alias(f"{event}_left_expectation_plus_9step"),
        )

        # 1.2 右側を計算(同じindexの予測を含まない右側を計算。逆順にして一個ずらしrolling_sumを取る必要がある）
        minute_pred_df = minute_pred_df.reverse()
        minute_pred_df = minute_pred_df.with_columns(
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .shift(1)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over(["series_id", "chunk_id"])
                        .fill_null(0)
                        / match_sums[i]
                    )
                    for i, window in enumerate(low_match_nums[1:])
                ]
            ).alias(f"{event}_right_expectation_plus_3step"),
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .shift(1)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over(["series_id", "chunk_id"])
                        .fill_null(0)
                        / match_sums[i]
                    )
                    for i, window in enumerate(high_match_nums[1:])
                ]
            ).alias(f"{event}_right_expectation_plus_9step"),
        )
        minute_pred_df = minute_pred_df.reverse()

        # 合計の期待値計算
        minute_pred_df = minute_pred_df.with_columns(
            (pl.col(f"{event}_left_expectation_plus_3step") + pl.col(f"{event}_right_expectation_plus_3step")).alias(
                f"{event}_expectation_sum_3step"
            ),
            (pl.col(f"{event}_left_expectation_plus_9step") + pl.col(f"{event}_right_expectation_plus_9step")).alias(
                f"{event}_expectation_sum_9step"
            ),
        )

        # print(display(minute_pred_df))

        # 3. 最大値の取得 & 期待値の割引
        """
        各予測地点の power を管理する。powerは以下の11種類
        0: その予測地点が影響を与える範囲は無い
        1: その予測地点が影響を与える範囲は左右1つ(1min)
        2: その予測地点が影響を与える範囲は左右3つ
        ︙
        10: 左右30(step 0~360)

        event を作るたびに、eventからtolerance内にある予測地点のpowerを下げる。
        その際に予測地点からtolerance内にある、eventがあったところも含めた候補地点の期待値を割り引く。
        """
        for series_id, series_df in tqdm(
            minute_pred_df.select(
                [
                    "series_id",
                    "chunk_id",
                    "step",
                    event_pred_col,
                    f"{event}_expectation_sum_3step",
                    f"{event}_expectation_sum_9step",
                ]
            ).group_by("series_id"),
            desc=f"detect {event} peaks",
            leave=False,
            total=len(minute_pred_df["series_id"].unique()),
            disable=tqdm_disable,
        ):
            # chunkごとの id の最大最小を計算
            series_df = series_df.with_row_count().with_columns(
                pl.col("row_nr").max().over(["chunk_id"]).alias("max_id_in_chunk"),
                pl.col("row_nr").min().over(["chunk_id"]).alias("min_id_in_chunk"),
            )

            preds = series_df[event_pred_col].to_numpy()
            expectation_sum_3step = series_df[f"{event}_expectation_sum_3step"].to_numpy(writable=True)
            expectation_sum_9step = series_df[f"{event}_expectation_sum_9step"].to_numpy(writable=True)
            steps = series_df[f"step"].to_numpy(writable=True)
            step_id_mins = series_df["min_id_in_chunk"].to_numpy(writable=True)
            step_id_maxs = series_df["max_id_in_chunk"].to_numpy(writable=True) + 1
            powers = np.ones(len(expectation_sum_3step), dtype=np.int32) * 10

            result_steps, result_scores = detect_events_for_serie(
                height,
                max_event_per_series,
                high_match_nums,
                low_match_nums,
                match_sums,
                steps,
                step_id_mins,
                step_id_maxs,
                preds,
                expectation_sum_3step,
                expectation_sum_9step,
                powers,
            )

            for i in range(len(result_steps)):
                result_events_records.append(
                    {
                        "series_id": series_id,
                        "step": result_steps[i],
                        "event": event,
                        "score": result_scores[i],
                    }
                )

            # print(expectation_sum_3step[max_step_index], expectation_sum_9step[max_step_index])

    if len(result_events_records) == 0:  # 一つも予測がない場合はdummyを入れる
        result_events_records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )
    sub_df = pl.DataFrame(result_events_records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df


@jit(nopython=True, cache=True)
def detect_events_for_serie(
    height,
    max_event_per_series,
    high_match_nums,
    low_match_nums,
    match_sums,
    steps,
    step_id_mins,
    step_id_maxs,
    preds,
    expectation_sum_3step,
    expectation_sum_9step,
    powers,
):
    result_steps = []
    result_scores = []
    for _ in range(max_event_per_series):  # 高い順に最大max_event_per_series個のeventを決定
        # 3.1 最大値の取得
        # 合計の期待値が最大のstepを取得
        max_step3 = expectation_sum_3step.argmax()
        max_score3 = expectation_sum_3step[max_step3]
        max_step9 = expectation_sum_9step.argmax()
        max_score9 = expectation_sum_9step[max_step9]
        if max_score3 > max_score9:
            # print('max_score3')
            left_nums = high_match_nums
            right_nums = low_match_nums
            max_step_index = max_step3
            max_score = max_score3
            if max_score < height:  # 閾値以下なら終了
                break
            result_steps.append(steps[max_step_index] + 3)
            result_scores.append(max_score)
        else:
            # print('max_score9')
            left_nums = low_match_nums
            right_nums = high_match_nums
            max_step_index = max_step9
            max_score = max_score9
            if max_score < height:  # 閾値以下なら終了
                break
            result_steps.append(steps[max_step_index] + 9)
            result_scores.append(max_score)
        # print(f"max_step_index:{max_step_index}, max_score:{max_score}")

        # 3.2 期待値の割引
        """
        各予測地点のpowerを修正するとともに、候補地点の期待値を割引く。
        powerが pi まで小さくなることによってその予測値が影響を与える範囲が狭くなる。
        つまり狭くなって範囲から抜けた expectation_sum の値が、その予測値の値*重みの分だけ小さくなる
        """
        # 3.2.1 まずはpowerを修正するstepの候補を探す
        target_step_powers = []  # (target_step, pred, base_power, power, step_min, step_max)のリスト
        for pi in range(0, 10):
            # 左側
            for l_diff in range(left_nums[pi], left_nums[pi + 1]):
                target_step_index = max_step_index - l_diff
                if target_step_index < 0:
                    break
                pred = preds[target_step_index]
                base_power = powers[target_step_index]
                if base_power > pi:  # power が小さくなる場合のみ修正
                    target_step_powers.append(
                        (
                            target_step_index,
                            pred,
                            base_power,
                            pi,
                            step_id_mins[target_step_index],
                            step_id_maxs[target_step_index],
                        )
                    )
            # 右側
            for r_diff in range(right_nums[pi] + 1, right_nums[pi + 1] + 1):  # 自分自身と同じindexは含めない
                target_step_index = max_step_index + r_diff
                if target_step_index >= len(powers):
                    break
                pred = preds[target_step_index]
                base_power = powers[target_step_index]
                if base_power > pi:
                    target_step_powers.append(
                        (
                            target_step_index,
                            pred,
                            base_power,
                            pi,
                            step_id_mins[target_step_index],
                            step_id_maxs[target_step_index],
                        )
                    )
        # print('target_step_powers', target_step_powers)

        # 3.2.2 対象となる step の power を修正するとともに期待値を割り引く
        """
        予測地点のpowerを下げるとともに、関連する候補地点の期待値を修正する。
        検出したeventから遠い予測地点の場合は、予測地点に近い候補地点であっても期待値はその分割り引かれる。
        3stepの修正をする時は target_stepから左側が low_match_nums, 右側が high_match_nums
        9stepの修正をする時は target_stepから左側が high_match_nums, 右側が low_match_nums
        だんだんと内側のみがのこるように修正する。

        - powerが 10 → 8 になるケースは左右1~30個に影響を及ぼしていたものが、左右の1~20個に影響を及ぼすようになる。また、powerが2個減った分全体の期待値も割り引かれる
        - powerが 10 → 5 になるケースは左右1~30個に影響を及ぼしていたものが、左右の1~12(13)個に影響を及ぼすようになる
        - powerが 8 → 7 になるケースは左右1~20個に影響を及ぼしていたものが、左右の1~15個に影響を及ぼすようになる
        """
        for si, pred, base_power, power, step_min, step_max in target_step_powers:
            # print(f"si:{si}, pred:{pred}, base_power:{base_power}, power:{power}")
            powers[si] = power
            # 中心ほど重みが強いので power ごとに処理
            for pi in range(base_power, power, -1):  # base_powerからpowerに減らしていくことで予測値の外側から削る
                # 3step
                left_diff_max = low_match_nums[pi]
                right_diff_max = high_match_nums[pi]
                expectation_sum_3step[max(si - left_diff_max, step_min) : min(si + right_diff_max, step_max)] -= (
                    pred / match_sums[pi - 1]
                )
                """
                if ((si-left_diff_max <= max_step_index) and (max_step_index < si+right_diff_max)):
                    print(f'3step pi:{pi}')
                    print(f"max_step_index:{max_step_index}, si:{si}, left_diff_max:{left_diff_max}, right_diff_max:{right_diff_max}") 
                    print("[", si-left_diff_max, si+right_diff_max, ")")
                    print(f"power: {pi}→{pi-1}, pred:{pred}")
                    print()
                """
                # 9step
                left_diff_max = high_match_nums[pi]
                right_diff_max = low_match_nums[pi]
                expectation_sum_9step[max(si - left_diff_max, step_min) : min(si + right_diff_max, step_max)] -= (
                    pred / match_sums[pi - 1]
                )
                """
                if ((si-left_diff_max <= max_step_index) and (max_step_index < si+right_diff_max)):
                    print(f'9step pi:{pi}')
                    print(f"max_step_index:{max_step_index}, si:{si}, left_diff_max:{left_diff_max}, right_diff_max:{right_diff_max}") 
                    print("[", si-left_diff_max, si+right_diff_max, ")")
                    print(f"power: {pi}→{pi-1}, pred:{pred}")
                    print()
                """
    return result_steps, result_scores


def remove_periodicity(
    df,
    periodicity_dict,
    event2col: dict[str, str] = {"onset": "prediction_onset", "wakeup": "prediction_wakeup"},
):
    """
    雑にdfからperiodicityを削除する
    """
    dfs = []
    for series_id, series_df in df.group_by("series_id"):
        for event, event_pred_col in event2col.items():
            series_df.with_columns(
                pl.Series(
                    name=event_pred_col,
                    values=series_df.get_column(event_pred_col).to_numpy() * (1 - periodicity_dict[series_id]),
                )
            )
        dfs.append(series_df)
    return pl.concat(dfs)


def post_process(
    pred_df,
    event_rate: int | float = 0.005,
    height: float = 0.1,
    event2col: dict[str, str] = {"onset": "prediction_onset", "wakeup": "prediction_wakeup"},
):
    """
    1分ごとの予測値を用いてイベントを検出する
    # TODO: 1段目のモデルを入れる場合は1分ごとの予測値をそのまま使わずに周辺の予測値をrollする方が良さそう？

    用語
    - 予測地点: 1段目 or 2段目のモデルによって得られた1分毎の予測位置
    - 候補地点: event の候補となる 15秒 or 45秒始まりの30秒間隔の位置

    Args:
        pred_df (pl.DataFrame): series_id, step, event2colのvalue をカラムに持つ。stepが12の倍数以外の行は無視される。periodicityは削除済み想定
        event_rate (int | float, optional): [0,1) の値であれば1分間に何回イベントが起こるか。intの場合はseries_idごとに同じイベント数を検出。 Defaults to 0.005.
        height (float, optional): 候補地点の期待値がこの値を下回ったら終了。 Defaults to 0.1.
    Returns:
        event_df (pl.DataFrame): row_id, series_id, step, event, score をカラムに持つ。
    """
    high_match_nums = [1, 3, 5, 8, 10, 13, 15, 20, 25, 30]
    low_match_nums = [1, 3, 5, 7, 10, 12, 15, 20, 25, 30]
    # match_sums = [2, 6, 10, 15, 20, 25, 30, 40, 50, 60]
    # total_num = sum(high_match_nums + low_match_nums)
    result_events_records = []

    # event ごとに処理
    for event, event_pred_col in event2col.items():
        """
        元の系列の予測地点(長さN): 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, ..., (N-1)*12
        15秒から30秒おきのevent候補地点(長さ2N): 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, ..., (N-1)*12+3, (N-1)*12+9
            - 15秒(3step)から1分おき(長さN): 3, 15, 27, 39, 51, 63, 75, 87, 99, 111, ..., (N-1)*12+3
                - 左の個数 {12: 1, 36: 3, 60: 5, 90: *8*, 120: 10, 150: *13*, 180: 15, 240: 20, 300: 25, 360: 30} high_match_nums
                - 右の個数 {12: 1, 36: 3, 60: 5, 90: *7*, 120: 10, 150: *12*, 180: 15, 240: 20, 300: 25, 360: 30} low_match_nums
            - 45秒(9step)から1分おき(長さN): 9, 21, 33, 45, 57, 69, 81, 93, 105, 117, ..., (N-1)*12+9
                - 左の個数 {12: 1, 36: 3, 60: 5, 90: *7*, 120: 10, 150: *12*, 180: 15, 240: 20, 300: 25, 360: 30} low_match_nums
                - 右の個数 {12: 1, 36: 3, 60: 5, 90: *8*, 120: 10, 150: *13*, 180: 15, 240: 20, 300: 25, 360: 30} high_match_nums       
        """

        # 12の倍数以外の行は無視して1分毎の予測値にする
        minute_pred_df = pred_df.filter(pl.col("step") % 12 == 0)
        max_event_per_series = event_rate if isinstance(event_rate, int) else int(len(minute_pred_df) * event_rate)

        # series_id, step でソート
        minute_pred_df = minute_pred_df.sort(["series_id", "step"])

        # 1. 期待値の計算
        # 1.1 左側を計算 (同じindexの予測を含む左側を計算)
        """
        以下をそれぞれ計算する
        - 15秒(3step)から1分おき(長さN)での候補地点での期待値: stepは 3, 15, 27, 39, 51, 63, 75, 87, 99, 111, ..., (N-1)*12+3
        - 45秒(9step)から1分おき(長さN)での候補地点での期待値: stepは 9, 21, 33, 45, 57, 69, 81, 93, 105, 117, ..., (N-1)*12+9
        計算は左側の予測地点の数と、右側の予測地点の数
        """
        minute_pred_df = minute_pred_df.with_columns(
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over("series_id")
                    )
                    for window in high_match_nums
                ]
            ).alias(f"{event}_left_expectation_plus_3step"),
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over("series_id")
                    )
                    for window in low_match_nums
                ]
            ).alias(f"{event}_left_expectation_plus_9step"),
        )

        # 1.2 右側を計算(同じindexの予測を含まない右側を計算。逆順にして一個ずらしrolling_sumを取る必要がある）
        minute_pred_df = minute_pred_df.reverse()
        minute_pred_df = minute_pred_df.with_columns(
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .shift(1)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over("series_id")
                        .fill_null(0)
                    )
                    for window in low_match_nums
                ]
            ).alias(f"{event}_right_expectation_plus_3step"),
            pl.sum_horizontal(
                [
                    (
                        pl.col(event_pred_col)
                        .shift(1)
                        .rolling_sum(window_size=window, center=False, min_periods=1)
                        .over("series_id")
                        .fill_null(0)
                    )
                    for window in high_match_nums
                ]
            ).alias(f"{event}_right_expectation_plus_9step"),
        )
        minute_pred_df = minute_pred_df.reverse()

        # 合計の期待値計算
        minute_pred_df = minute_pred_df.with_columns(
            (pl.col(f"{event}_left_expectation_plus_3step") + pl.col(f"{event}_right_expectation_plus_3step")).alias(
                f"{event}_expectation_sum_3step"
            ),
            (pl.col(f"{event}_left_expectation_plus_9step") + pl.col(f"{event}_right_expectation_plus_9step")).alias(
                f"{event}_expectation_sum_9step"
            ),
        )

        # 3. 最大値の取得 & 期待値の割引
        """
        各予測地点の power を管理する。powerは以下の11種類
        0: その予測地点が影響を与える範囲は無い
        1: その予測地点が影響を与える範囲は左右1つ(1min)
        2: その予測地点が影響を与える範囲は左右3つ
        ︙
        10: 左右30(step 0~360)

        event を作るたびに、eventからtolerance内にある予測地点のpowerを下げる。
        その際に予測地点からtolerance内にある、eventがあったところも含めた候補地点の期待値を割り引く。
        """
        for series_id, series_df in tqdm(
            minute_pred_df.select(
                ["series_id", event_pred_col, f"{event}_expectation_sum_3step", f"{event}_expectation_sum_9step"]
            ).group_by("series_id"),
            desc="find peaks",
            leave=False,
            total=len(minute_pred_df["series_id"].unique()),
        ):
            preds = series_df[event_pred_col].to_numpy()
            expectation_sum_3step = series_df[f"{event}_expectation_sum_3step"].to_numpy(writable=True)
            expectation_sum_9step = series_df[f"{event}_expectation_sum_9step"].to_numpy(writable=True)
            powers = np.ones(len(expectation_sum_3step), dtype=np.int32) * 10
            for _ in range(max_event_per_series):  # 高い順に最大max_event_per_series個のeventを決定
                # 3.1 最大値の取得
                # 合計の期待値が最大のstepを取得
                max_step3 = expectation_sum_3step.argmax()
                max_score3 = expectation_sum_3step[max_step3]
                max_step9 = expectation_sum_9step.argmax()
                max_score9 = expectation_sum_9step[max_step9]
                if max_score3 > max_score9:
                    # print('max_score3')
                    left_nums = [0] + high_match_nums
                    right_nums = [0] + low_match_nums
                    max_step_index = max_step3
                    max_score = max_score3
                    result_events_records.append(
                        {
                            "series_id": series_id,
                            "step": max_step_index * 12 + 3,
                            "event": event,
                            "score": max_score,
                        }
                    )
                else:
                    # print('max_score9')
                    left_nums = [0] + low_match_nums
                    right_nums = [0] + high_match_nums
                    max_step_index = max_step9
                    max_score = max_score9
                    result_events_records.append(
                        {
                            "series_id": series_id,
                            "step": max_step_index * 12 + 9,
                            "event": event,
                            "score": max_score,
                        }
                    )
                if max_score < height:  # 閾値以下なら終了
                    break
                # print(f"max_step_index:{max_step_index}, max_score:{max_score}")

                # 3.2 期待値の割引
                """
                各予測地点のpowerを修正するとともに、候補地点の期待値を割引く。
                powerが pi まで小さくなることによってその予測値が影響を与える範囲が狭くなる。
                つまり狭くなって範囲から抜けた expectation_sum の値が、その予測値の値*重みの分だけ小さくなる
                """
                # 3.2.1 まずはpowerを修正するstepの候補を探す
                target_step_powers = []  # (target_step, pred, base_power, power)のリスト
                for pi in range(0, 10):
                    # 左側
                    for l_diff in range(left_nums[pi], left_nums[pi + 1]):
                        target_step_index = max_step_index - l_diff
                        if target_step_index < 0:
                            break
                        pred = preds[target_step_index]
                        base_power = powers[target_step_index]
                        if base_power > pi:  # power が小さくなる場合のみ修正
                            target_step_powers.append((target_step_index, pred, base_power, pi))
                    # 右側
                    for r_diff in range(right_nums[pi] + 1, right_nums[pi + 1] + 1):  # 自分自身と同じindexは含めない
                        target_step_index = max_step_index + r_diff
                        if target_step_index >= len(powers):
                            break
                        pred = preds[target_step_index]
                        base_power = powers[target_step_index]
                        if base_power > pi:
                            target_step_powers.append((target_step_index, pred, base_power, pi))
                # print('target_step_powers', target_step_powers)

                # 3.2.2 対象となる step の power を修正するとともに期待値を割り引く
                """
                予測地点のpowerを下げるとともに、関連する候補地点の期待値を修正する。
                検出したeventから遠い予測地点の場合は、予測地点に近い候補地点であっても期待値はその分割り引かれる。
                
                3stepの修正をする時は target_stepから左側が low_match_nums, 右側が high_match_nums
                9stepの修正をする時は target_stepから左側が high_match_nums, 右側が low_match_nums
                だんだんと内側のみがのこるように修正する。

                - powerが 10 → 8 になるケースは左右1~30個に影響を及ぼしていたものが、左右の1~20個に影響を及ぼすようになる。また、powerが2個減った分全体の期待値も割り引かれる
                - powerが 10 → 5 になるケースは左右1~30個に影響を及ぼしていたものが、左右の1~12(13)個に影響を及ぼすようになる
                - powerが 8 → 7 になるケースは左右1~20個に影響を及ぼしていたものが、左右の1~15個に影響を及ぼすようになる
                """
                # print(expectation_sum_3step[max_step_index], expectation_sum_9step[max_step_index])
                for si, pred, base_power, power in target_step_powers:
                    # print(f"si:{si}, pred:{pred}, base_power:{base_power}, power:{power}")
                    powers[si] = power
                    # 中心ほど重みが強いので power ごとに処理
                    for pi in range(
                        base_power, power, -1
                    ):  # base_powerからpowerに減らしていくことで予測値の外側から削る
                        # 3step
                        left_nums = [0] + low_match_nums
                        right_nums = [0] + high_match_nums
                        left_diff_max = left_nums[pi]
                        right_diff_max = right_nums[pi]
                        expectation_sum_3step[si - left_diff_max : si + right_diff_max] -= pred
                        """
                        if ((si-left_diff_max <= max_step_index) and (max_step_index < si+right_diff_max)):
                            print(f'3step pi:{pi}')
                            print(f"max_step_index:{max_step_index}, si:{si}, left_diff_max:{left_diff_max}, right_diff_max:{right_diff_max}") 
                            print("[", si-left_diff_max, si+right_diff_max, ")")
                            print(f"power: {pi}→{pi-1}, pred:{pred}")
                            print()
                        """

                        # 9step
                        left_nums = [0] + high_match_nums
                        right_nums = [0] + low_match_nums
                        left_diff_max = left_nums[pi]
                        right_diff_max = right_nums[pi]
                        expectation_sum_9step[si - left_diff_max : si + right_diff_max] -= pred
                        """
                        if ((si-left_diff_max <= max_step_index) and (max_step_index < si+right_diff_max)):
                            print(f'9step pi:{pi}')
                            print(f"max_step_index:{max_step_index}, si:{si}, left_diff_max:{left_diff_max}, right_diff_max:{right_diff_max}") 
                            print("[", si-left_diff_max, si+right_diff_max, ")")
                            print(f"power: {pi}→{pi-1}, pred:{pred}")
                            print()
                        """

                # print(expectation_sum_3step[max_step_index], expectation_sum_9step[max_step_index])

    if len(result_events_records) == 0:  # 一つも予測がない場合はdummyを入れる
        result_events_records.append(
            {
                "series_id": series_id,
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )
    sub_df = pl.DataFrame(result_events_records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df
