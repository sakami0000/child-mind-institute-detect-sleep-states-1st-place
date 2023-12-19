import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import uniform_filter1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")
LOGGER = logging.getLogger(Path(__file__).name)


def downsample(sequence, factor=10):
    """
    Downsamples the sequence by the given factor.
    """
    return sequence[::factor]


def resize_1d_array(array, new_size):
    """
    Resizes a 1D numpy array to a new size using interpolation.
    """
    return np.interp(np.linspace(0, len(array) - 1, new_size), np.arange(len(array)), array)


def compare_rows_floating(matrix: np.ndarray, atol=1e-8) -> np.ndarray:
    """
    matrix (n, d) に対して、各行のベクトル同士の比較を行い、同じベクトルがあれば True とする (n, n) の行列を返す
    """
    comparison_matrix = np.all(
        np.isclose(matrix[:, None, :], matrix[None, :, :], atol=atol), axis=2
    )
    return comparison_matrix


def predict_periodicity(
    seq: np.ndarray, downsample_rate: int = 15, split_hour: int = 8, th=0.99, how="comp"
) -> np.ndarray:
    """
    split_hourごとにフレームに分割して、同じ波形が現れたフレームは周期性ありとみなす。
    Args:
        seq (np.ndarray): 1D array of shape (n,)
        downsample_rate (int, optional): 小さいほど間違ったものを検出しにくくなるが遅くなる
        split_hour (int, optional):  周期性の検出期間は split_hour ごとに行われる。24 の約数であること。大きいほど間違ったものを検出しにくくなる。
        how (str, optional): "comp" なら要素ごとに比較(計算量が多い)、"sim" なら cos類似度を使う
    Returns:
        pred (np.ndarray): 1D array of shape (n,)
    """
    # 最低限必要な長さがなければ、周期性はないとみなす
    if len(seq) < 24 * 3600 // 5:
        return np.zeros(len(seq), dtype=bool)

    # seq をダウンサンプリングして seq_downsampled に
    seq_downsampled = downsample(seq, downsample_rate)

    # seq_downsampled を split_hour ごとに分割した chunks (chunk_num, d) を作る（足りない部分は0埋め）
    split_step = split_hour * 3600 // 5 // downsample_rate
    valid_length = (
        (len(seq_downsampled) + (split_step - 1)) // split_step
    ) * split_step  # split_step に合うように
    seq_downsampled_padded = np.zeros(valid_length)
    seq_downsampled_padded[: len(seq_downsampled)] = seq_downsampled
    chunks = seq_downsampled_padded.reshape(-1, split_step)

    if how == "comp":
        # chunk_num サイズの予測 pred_chunk を得る
        sim_matrix = compare_rows_floating(chunks)
        sim_matrix[range(len(sim_matrix)), range(len(sim_matrix))] = 0  # 対角要素を０に
        pred_chunk = sim_matrix.max(axis=0)
    elif how == "sim":
        # 各ベクトルを正規化し chunks・chunks.T で (chunk_num,chunk_num) のcos類似度を求め、対角線上を0にした後にmaxを取って chunk_num サイズの予測 pred_chunk を得る
        norm_vecs = chunks / np.linalg.norm(chunks, axis=1, keepdims=True)
        cosine_sim_matrix = np.dot(norm_vecs, norm_vecs.T)
        cosine_sim_matrix[range(len(cosine_sim_matrix)), range(len(cosine_sim_matrix))] = 0
        pred_chunk = cosine_sim_matrix.max(axis=0) > th

    # 最後の一個前が true なら、最後もtrueにする（最後は0埋めしたのでうまくできていない）
    pred_chunk[-1] = pred_chunk[-2:-1].max()

    # pred_vecを元のsequenceのサイズに戻す
    pred = resize_1d_array(pred_chunk.repeat(split_step)[: len(seq_downsampled)], len(seq))
    return pred


def predict_periodicity_v2(
    seq: np.ndarray, downsample_rate: int = 12, stride_min: int = 3, split_min: int = 360
) -> np.ndarray:
    """
    split_hourごとにフレームに分割して、同じ波形が現れたフレームは周期性ありとみなす。開始位置は stride_min ずつずらして行い、結果は max で集約する。

    Args:
        seq (np.ndarray): 1D array of shape (n,)
        downsample_rate (int, optional): 小さいほど間違ったものを検出しにくくなるが遅くなる
        split_hour (int, optional):  周期性の検出期間は split_hour ごとに行われる。24 の約数であること。大きいほど間違ったものを検出しにくくなる。
    Returns:
        pred (np.ndarray): 1D array of shape (n,)
    """

    seq = (seq - seq.mean()) / seq.std()
    pred = np.zeros(len(seq), dtype=np.float32)

    # 最低限必要な長さがなければ、周期性はないとみなす
    if len(seq) < 24 * 3600 // 5:
        return pred

    stride_step = stride_min * 12
    split_step = split_min * 12

    for start_step in range(0, split_step, stride_step):
        tmp_pred = predict_periodicity(
            seq[start_step:],
            downsample_rate=downsample_rate,
            split_hour=split_min // 60,
            how="comp",
        )
        pred[start_step:] = np.maximum(pred[start_step:], tmp_pred)

    return pred


def get_periodicity_dict(cfg: DictConfig) -> dict[np.ndarray]:
    phase = cfg.phase if "phase" in cfg else "train"
    feature_dir = Path(cfg.dir.processed_dir) / phase
    LOGGER.info(f"feature_dir: {feature_dir}")
    series_ids = [x.name for x in feature_dir.glob("*")]
    periodicity_dict = {}
    for series_id in series_ids:
        periodicity = np.load(feature_dir / series_id / "periodicity.npy")
        periodicity = np.minimum(
            periodicity,
            uniform_filter1d(periodicity, size=cfg.post_process.periodicity.filter_size),
        )
        periodicity_dict[series_id] = periodicity

        """
        seq = np.load(feature_dir / series_id / "enmo.npy")
        periodicity_dict[series_id] = predict_periodicity(seq, cfg.post_process.periodicity.downsample_rate, cfg.post_process.periodicity.split_hour, cfg.post_process.periodicity.th)
        """
    LOGGER.info(f"series length: {len(series_ids)}")
    return periodicity_dict
