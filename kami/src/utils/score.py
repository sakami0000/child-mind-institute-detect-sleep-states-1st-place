import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg, post_process_for_seg_group_by_day


def score_group_by_day(val_event_df: pl.DataFrame, keys: list[str], preds: np.ndarray, val_df: pl.DataFrame) -> float:
    """
    日毎に最大値のeventを検出し、それをsubmissionとしてスコアリングする

    Args:
        val_event_df (pl.DataFrame): Ground truth
        keys (list[str]): 予測したchunkのkey({series_id}_{chunk_id})
        preds (np.ndarray): (chunk_num, duration, 2)
        val_df (pl.DataFrame): sequence
    """
    submission_df = post_process_for_seg_group_by_day(keys, preds, val_df)
    score = event_detection_ap(
        val_event_df.to_pandas(),
        submission_df.to_pandas(),
    )
    return score


def score_ternary_search_distance(
    val_event_df: pl.DataFrame, keys: list[str], preds: np.ndarray, score_th: float = 0.005
) -> [float, float]:
    """
    post_process_for_seg のパラメータdistanceを ternary searchで探索する
    """
    l = 5
    r = 100

    cnt = 0
    best_score = 0.0
    best_distance = 0

    for cnt in tqdm(range(30)):
        if r - l < 1:
            break
        m1 = int(l + (r - l) / 3)
        m2 = int(r - (r - l) / 3)
        score1 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=score_th,
                distance=m1,
            ).to_pandas(),
        )
        score2 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=score_th,
                distance=m2,
            ).to_pandas(),
        )

        if score1 >= score2:
            r = m2
            best_score = score1
            best_distance = m1

        else:
            l = m1
            best_score = score2
            best_distance = m2

        tqdm.write(f"score1(m1): {score1:.5f}({m1:.5f}), score2(m2): {score2:.5f}({m2:.5f}), l: {l:.5f}, r: {r:.5f}")

        if abs(m2 - m1) <= 2:
            break

    return best_score, best_distance


def score_ternary_search_th(
    val_event_df: pl.DataFrame, keys: list[str], preds: np.ndarray, distance: int = 5000
) -> [float, float]:
    """
    post_process_for_seg のパラメータ score_th を ternary searchで探索する
    """
    l = 0.0
    r = 1.0

    cnt = 0
    best_score = 0.0
    best_th = 0.0

    for cnt in tqdm(range(30)):
        if r - l < 0.01:
            break
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        score1 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=m1,
                distance=distance,
            ).to_pandas(),
        )
        score2 = event_detection_ap(
            val_event_df.to_pandas(),
            post_process_for_seg(
                keys=keys,
                preds=preds,
                score_th=m2,
                distance=distance,
            ).to_pandas(),
        )
        if score1 >= score2:
            r = m2
            best_score = score1
            best_th = m1
        else:
            l = m1
            best_score = score2
            best_th = m2

        tqdm.write(f"score1(m1): {score1:.5f}({m1:.5f}), score2(m2): {score2:.5f}({m2:.5f}), l: {l:.5f}, r: {r:.5f}")

    return best_score, best_th
