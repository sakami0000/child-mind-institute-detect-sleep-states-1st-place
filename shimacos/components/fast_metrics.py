"""Event Detection Average Precision

An average precision metric for event detection in time series and
video.

"""

from bisect import bisect_left
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class ParticipantVisibleError(Exception):
    pass


# Set some placeholders for global parameters
series_id_column_name = "series_id"
time_column_name = "step"
event_column_name = "event"
score_column_name = "score"
use_scoring_intervals = False
tolerances = {
    "onset": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    "wakeup": [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}

commoon_tolerances = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]],
    series_id_column_name: str,
    time_column_name: str,
    event_column_name: str,
    score_column_name: str,
    use_scoring_intervals: bool = False,
) -> float:
    # Validate metric parameters
    assert len(tolerances) > 0, "Events must have defined tolerances."
    assert set(tolerances.keys()) == set(solution[event_column_name]).difference(
        {"start", "end"}
    ), (
        f"Solution column {event_column_name} must contain the same events "
        "as defined in tolerances."
    )
    assert pd.api.types.is_numeric_dtype(
        solution[time_column_name]
    ), f"Solution column {time_column_name} must be of numeric type."

    # Validate submission format
    for column_name in [
        series_id_column_name,
        time_column_name,
        event_column_name,
        score_column_name,
    ]:
        if column_name not in submission.columns:
            raise ParticipantVisibleError(f"Submission must have column '{column_name}'.")

    if not pd.api.types.is_numeric_dtype(submission[time_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{time_column_name}' must be of numeric type."
        )
    if not pd.api.types.is_numeric_dtype(submission[score_column_name]):
        raise ParticipantVisibleError(
            f"Submission column '{score_column_name}' must be of numeric type."
        )

    # Set these globally to avoid passing around a bunch of arguments
    globals()["series_id_column_name"] = series_id_column_name
    globals()["time_column_name"] = time_column_name
    globals()["event_column_name"] = event_column_name
    globals()["score_column_name"] = score_column_name
    globals()["use_scoring_intervals"] = use_scoring_intervals

    return event_detection_ap(solution, submission, tolerances)


def event_detection_ap(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    tolerances: Dict[str, List[float]] = tolerances,  # type: ignore
    with_table: bool = False,
) -> float | Tuple[float, pd.DataFrame]:
    # Ensure solution and submission are sorted properly
    solution = solution.sort_values([series_id_column_name, time_column_name])
    submission = submission.sort_values([series_id_column_name, time_column_name])

    # Extract scoring intervals.
    if use_scoring_intervals:
        # intervals = (
        #     solution.query("event in ['start', 'end']")
        #     .assign(
        #       interval=lambda x: x.groupby([series_id_column_name, event_column_name]).cumcount()
        #     )
        #     .pivot(
        #         index="interval",
        #         columns=[series_id_column_name, event_column_name],
        #         values=time_column_name,
        #     )
        #     .stack(series_id_column_name)
        #     .swaplevel()
        #     .sort_index()
        #     .loc[:, ["start", "end"]]
        #     .apply(lambda x: pd.Interval(*x, closed="both"), axis=1)
        # )
        pass

    # Extract ground-truth events.
    ground_truths = solution.query("event not in ['start', 'end']").reset_index(drop=True)

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts(event_column_name).to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched=False)

    # Remove detections outside of scoring intervals
    if use_scoring_intervals:
        # detections_filtered = []
        # for (det_group, dets), (int_group, ints) in zip(
        #     detections.groupby(series_id_column_name), intervals.groupby(series_id_column_name)
        # ):
        #     assert det_group == int_group
        #     detections_filtered.append(filter_detections(dets, ints))  # noqa: F821
        # detections_filtered = pd.concat(detections_filtered, ignore_index=True)
        pass
    else:
        detections_filtered = detections

    aggregation_keys = pd.DataFrame(
        [
            (ev, vid)
            for ev in tolerances.keys()
            for vid in ground_truths[series_id_column_name].unique()
        ],
        columns=[event_column_name, series_id_column_name],
    )

    # Create match evaluation groups: event-class x tolerance x series_id → event-class x  series_id
    detections_grouped = aggregation_keys.merge(
        detections_filtered, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, series_id_column_name])
    ground_truths_grouped = aggregation_keys.merge(
        ground_truths, on=[event_column_name, series_id_column_name], how="left"
    ).groupby([event_column_name, series_id_column_name])

    # Match detections to ground truth events by evaluation group
    detections_matched = []

    for key, dets in tqdm(
        detections_grouped,
        dynamic_ncols=True,
        leave=False,
        desc="Matching detections to ground truth events",
    ):  # get_groupは時間がかかるので要素数の多いやつをfor文で取得
        gts = ground_truths_grouped.get_group(key)
        detections_matched.append(match_detections(commoon_tolerances, gts, dets))

    detections_matched = pd.concat(detections_matched)

    # Compute AP per event
    event_classes = ground_truths[event_column_name].unique()

    ap_results = []
    for tol in commoon_tolerances:
        ap_df = (
            detections_matched.query("event in @event_classes")
            .groupby([event_column_name])
            .apply(
                lambda group: average_precision_score_with(
                    group[f"matched_{tol}"].to_numpy(),
                    group[score_column_name].to_numpy(),
                    class_counts[group[event_column_name].iat[0]],
                )
            )
        )
        ap_df["tolerance"] = tol
        ap_results.append(ap_df)
    ap_table = pd.concat(ap_results)
    ap_table = ap_table.reset_index(drop=False).set_index([event_column_name, "tolerance"])

    # Average over tolerances, then over event classes
    mean_ap = ap_table.groupby(event_column_name)["ap"].mean().sum() / len(event_classes)

    if with_table:
        return mean_ap, ap_table
    else:
        return mean_ap


def find_nearest_time_idx(times, target_time, excluded_indices_dict, tolerances):
    """Find the index of the nearest time to the target_time
    that is not in excluded_indices."""
    idx = bisect_left(times, target_time)

    best_idx_dict = dict([(tol, None) for tol in tolerances])
    best_error_dict = dict([(tol, float("inf")) for tol in tolerances])

    offset_range = min(len(times), 3)  # 基本的には左右数個のGTを確認すれば十分。本当は1個でも良いはず
    for offset in range(-offset_range, offset_range):  # Check the exact, one before, and one after
        check_idx = idx + offset
        if 0 <= check_idx < len(times):
            for tol in tolerances:
                if check_idx not in excluded_indices_dict[tol]:
                    error = abs(times[check_idx] - target_time)
                    if error < best_error_dict[tol]:
                        best_error_dict[tol] = error
                        best_idx_dict[tol] = check_idx
    return best_idx_dict, best_error_dict


def match_detections(
    tolerances: list[int], ground_truths: pd.DataFrame, detections: pd.DataFrame
) -> pd.DataFrame:
    detections_sorted = detections.sort_values(score_column_name, ascending=False).dropna()
    is_matched = [
        np.full_like(detections_sorted[event_column_name], False, dtype=bool)
        for i in range(len(tolerances))
    ]
    ground_truths_times = ground_truths.sort_values(time_column_name)[time_column_name].tolist()
    matched_gt_indices_dict: dict[int, set[int]] = dict([(tol, set()) for tol in tolerances])

    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        det_time = getattr(det, time_column_name)

        best_idx_dict, best_error_dict = find_nearest_time_idx(
            ground_truths_times, det_time, matched_gt_indices_dict, tolerances
        )

        for ti, tol in enumerate(tolerances):
            if best_idx_dict[tol] is not None and best_error_dict[tol] < tol:
                is_matched[ti][i] = True
                matched_gt_indices_dict[tol].add(best_idx_dict[tol])

    for ti, tol in enumerate(tolerances):
        detections_sorted[f"matched_{tol}"] = is_matched[ti]
    return detections_sorted


def precision_recall_curve(
    matches: np.ndarray, scores: np.ndarray, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(matches) == 0:
        return [1], [0], []  # type: ignore

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind="stable")[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]
    fps = np.cumsum(~matches)[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = (
        tps / p
    )  # total number of ground truths might be different than total number of matches

    # Stop when full recall attained and reverse the outputs so recall is non-increasing.
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    # Final precision is 1 and final recall is 0
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])


def average_precision_score_with(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
    precision, recall, _ = precision_recall_curve(matches, scores, p)
    # Compute step integral
    return pd.Series(
        {
            "ap": -np.sum(np.diff(recall) * np.array(precision)[:-1]),
            "precision": precision,
            "recall": recall,
        }
    )
