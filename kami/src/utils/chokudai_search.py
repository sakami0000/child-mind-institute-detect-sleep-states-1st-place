import numpy as np
import polars as pl
from tqdm.auto import tqdm
from numba import jit
import copy


def chokudai_search_from_2nd(
    pred_df,
    max_event_per_date: int = 50,
    event2col: dict[str, str] = {"onset": "stacking_prediction_onset", "wakeup": "stacking_prediction_wakeup"},
    event2offset: dict[str, str] = {"onset": "5h", "wakeup": "0h"},
    use_daily_norm: bool = True,
    daily_score_offset=10,
    later_date_max_sub_rate: float | None = 0.05,
    time_limit: int = 1,
    candidate_num: int = 3,
) -> pl.DataFrame:
    """
    1分ごとの予測値を用いてイベントを検出する
    用語
    - 予測地点: 2段目のモデルによって得られた1分毎の予測位置
    - 候補地点: event の候補となる 15秒 or 45秒始まりの30秒間隔の位置

    Args:
        pred_df (pl.DataFrame): timestamp 込み
        event_rate (int | float, optional): [0,1) の値であれば1分間に何回イベントが起こるか。intの場合はseries_idごとに同じイベント数を検出。 Defaults to 0.005.
        height (float, optional): 候補地点の期待値がこの値を下回ったら終了。 Defaults to 0.1.
        time_limit (int, optional): chokudai search １回あたりの時間制限。 Defaults to 1.
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
        # 3. date ごとに期待値が最大となるように event を検出
        for (series_id, date), series_df in tqdm(
            minute_pred_df.select(
                [
                    "series_id",
                    "chunk_id",
                    "step",
                    event_pred_col,
                    f"{event}_expectation_sum_3step",
                    f"{event}_expectation_sum_9step",
                    "date",
                ]
            ).group_by(["series_id", "date"]),
            desc=f"detect {event} peaks",
            leave=False,
            total=len(minute_pred_df[["series_id", "date"]].unique()),
        ):
            # chunkごとの id の最大最小を計算
            series_df = series_df.with_row_count().with_columns(
                pl.col("row_nr").max().over(["chunk_id", "date"]).alias("max_id_in_chunk"),
                pl.col("row_nr").min().over(["chunk_id", "date"]).alias("min_id_in_chunk"),
            )

            preds = series_df[event_pred_col].to_numpy()
            expectation_sum_3step = series_df[f"{event}_expectation_sum_3step"].to_numpy(writable=True)
            expectation_sum_9step = series_df[f"{event}_expectation_sum_9step"].to_numpy(writable=True)
            steps = series_df[f"step"].to_numpy(writable=True)
            step_id_mins = series_df["min_id_in_chunk"].to_numpy(writable=True)
            step_id_maxs = series_df["max_id_in_chunk"].to_numpy(writable=True) + 1

            result_steps, result_scores = chokudai_search_for_one_day(
                steps,
                preds,
                step_id_mins,
                step_id_maxs,
                expectation_sum_3step,
                expectation_sum_9step,
                time_limit=time_limit,
                max_event=max_event_per_date,
                chokudai_width=1,
                candidate_num=candidate_num,
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


from numba import jit, objmode

# chokudai search
import heapq
import time


class State:
    def __init__(
        self,
        sum_score: float,
        steps: list[int],
        scores: list[float],
        expectation_sum_3step: np.ndarray,
        expectation_sum_9step: np.ndarray,
        powers: np.ndarray,
    ):
        self.sum_score = sum_score
        self.steps = steps
        self.scores = scores
        self.expectation_sum_3step = expectation_sum_3step
        self.expectation_sum_9step = expectation_sum_9step
        self.powers = powers

    def __lt__(self, other):
        return self.sum_score > other.sum_score

    def get_k_best_next_states(
        self,
        steps,
        preds,
        step_id_mins,
        step_id_maxs,
        candidate_num: int = 3,
    ) -> list["State"]:
        """
        このstateからcandidate_num個の次のstateを返す
        """
        next_states = []
        top_k_step_id_3 = np.argpartition(-self.expectation_sum_3step, candidate_num)[:candidate_num]
        top_k_step_id_9 = np.argpartition(-self.expectation_sum_9step, candidate_num)[:candidate_num]
        max_step_ids = []

        # 候補を追加 (expectation_sum_3/9step, powers だけはコピーはされるが更新はされていない)
        for step_id_3 in top_k_step_id_3:
            score = self.expectation_sum_3step[step_id_3]
            new_state = copy.deepcopy(self)
            new_state.sum_score += score
            new_state.steps.append(steps[step_id_3] + 3)
            new_state.scores.append(score)
            next_states.append(new_state)
            max_step_ids.append(step_id_3)
        for step_id_9 in top_k_step_id_9:
            score = self.expectation_sum_9step[step_id_9]
            new_state = copy.deepcopy(self)
            new_state.sum_score += score
            new_state.steps.append(steps[step_id_9] + 9)
            new_state.scores.append(score)
            next_states.append(new_state)
            max_step_ids.append(step_id_9)

        # 期待値の割引(expectation_sum を更新)
        for i, next_state in enumerate(next_states):
            update_state(
                preds,
                step_id_mins,
                step_id_maxs,
                max_step_ids[i],
                next_state.steps[-1],
                next_state.expectation_sum_3step,
                next_state.expectation_sum_9step,
                next_state.powers,
            )

        return next_states


@jit(nopython=True, cache=True)
def update_state(
    preds,
    step_id_mins,
    step_id_maxs,
    max_step_index,
    step,
    expectation_sum_3step,
    expectation_sum_9step,
    powers,
):
    high_match_nums = (0, 1, 3, 5, 8, 10, 13, 15, 20, 25, 30)
    low_match_nums = (0, 1, 3, 5, 7, 10, 12, 15, 20, 25, 30)

    if step % 12 == 3:
        left_nums = high_match_nums
        right_nums = low_match_nums
    else:
        left_nums = low_match_nums
        right_nums = high_match_nums

    target_step_powers = []  # (target_step, pred, base_power, new_power, step_min, step_max)のリスト
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
    for si, pred, base_power, new_power, step_min, step_max in target_step_powers:
        # print(f"si:{si}, pred:{pred}, base_power:{base_power}, power:{power}")
        powers[si] = new_power
        # 中心ほど重みが強いので power ごとに処理
        for pi in range(base_power, new_power, -1):  # base_powerからpowerに減らしていくことで予測値の外側から削る
            # 3step
            left_diff_max = low_match_nums[pi]
            right_diff_max = high_match_nums[pi]
            expectation_sum_3step[max(si - left_diff_max, step_min) : min(si + right_diff_max, step_max)] -= pred
            # 9step
            left_diff_max = high_match_nums[pi]
            right_diff_max = low_match_nums[pi]
            expectation_sum_9step[max(si - left_diff_max, step_min) : min(si + right_diff_max, step_max)] -= pred


def chokudai_search_for_one_day(
    steps,
    preds,
    step_id_mins,
    step_id_maxs,
    expectation_sum_3step: np.ndarray,
    expectation_sum_9step: np.ndarray,
    time_limit: int,
    max_event: int,
    chokudai_width: int = 1,
    candidate_num: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    一日分のデータに対してchokudai searchを行う

    Returns:
        tuple[np.ndarray, np.ndarray]: (steps, scores) のタプル
    """
    max_turn = max_event

    # 初期化
    heap_states = [[] for _ in range(max_turn + 1)]
    heap_states[0].append(
        State(
            0.0,
            [],
            [],
            expectation_sum_3step.copy(),
            expectation_sum_9step.copy(),
            powers=np.ones(len(expectation_sum_3step), dtype=np.int32) * 10,
        )
    )

    # time_limitの間だけ実行する
    start_time = time.time()  # 計測開始
    while time.time() - start_time < time_limit:
        for turn_i in range(max_turn):
            for _ in range(chokudai_width):
                if len(heap_states[turn_i]) == 0:
                    break
                state = heapq.heappop(heap_states[turn_i])
                for next_state in state.get_k_best_next_states(
                    steps,
                    preds,
                    step_id_mins,
                    step_id_maxs,
                    candidate_num=candidate_num,
                ):
                    heapq.heappush(heap_states[turn_i + 1], next_state)

    # 最大のスコアを持つものを返す
    best_state = heapq.heappop(heap_states[max_turn])

    return best_state.steps, best_state.scores
