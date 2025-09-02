import numpy as np
import librosa


LATENT_DIM = 64
COND_DIM = 64
MEL_BINS = 128
TIME_STEPS = 512
SR = 44100


def wav_to_mel(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y, sr=SR, n_fft=2048, hop_length=256, n_mels=MEL_BINS
    )
    mel = np.log1p(mel).astype("float32")
    mel = mel[:TIME_STEPS, :]
    return mel[None, :, :, None]


def mel_to_wav(mel):
    mel = np.expm1(mel[0, :, :, 0])
    wav = librosa.feature.inverse.mel_to_audio(
        mel, sr=SR, n_fft=2048, hop_length=256
    )
    return wav


def slice_into_windows(mel_time_nmel, win_steps=512, step_stride=512):
    """
    mel_time_nmel: 形状 (T, n_mels)  ※ librosa出力( n_mels, T )を転置しておく
    win_steps:     1ウィンドウのフレーム数
    step_stride:   次の窓へ進むステップ（オーバーラップなし=win_steps）
    return: list of (win_steps, n_mels)
    """
    T = mel_time_nmel.shape[0]
    out = []
    for start in range(0, max(1, T - win_steps + 1), step_stride):
        out.append(mel_time_nmel[start: start + win_steps, :])
    # 端をパディングしてでも入れたい場合
    if T % step_stride != 0 and T > win_steps:
        tail = mel_time_nmel[-win_steps:, :]
        out.append(tail)
    return out  # 学習時はランダム開始位置1個をサンプリングでもOK


def hann_fade(n):
    # 半ハン窓（0→1 or 1→0）を作る補助
    x = np.linspace(0, np.pi, n)
    return np.sin(x / 2) ** 2  # 0→1


def overlap_add_mel(chunks, hop_steps, fade_steps, n_mels):
    """
    chunks: List[np.ndarray] 各 (win_steps, n_mels) の **生成差分メル** など
    hop_steps: 次チャンクの開始オフセット(フレーム)
    fade_steps: オーバーラップ領域のクロスフェード幅(フレーム)
    n_mels: メル次元
    return: (T_total, n_mels)
    """
    if not chunks:
        return np.zeros((0, n_mels), dtype=np.float32)

    win_steps = chunks[0].shape[0]
    # 総フレーム長を推定
    T_total = hop_steps * (len(chunks) - 1) + win_steps
    out = np.zeros((T_total, n_mels), dtype=np.float32)
    weight = np.zeros((T_total, 1), dtype=np.float32)

    fade_in = hann_fade(fade_steps)[:, None]  # (fade_steps, 1)

    for i, ch in enumerate(chunks):
        start = i * hop_steps
        end = start + win_steps
        # 本体
        out[start:end, :] += ch
        weight[start:end, :] += 1.0

        # オーバーラップする部分にクロスフェードを適用
        if i > 0 and fade_steps > 0:
            ov_start = start
            ov_end = start + fade_steps
            # 前回分を fade_out、今回分を fade_in で重みづけ
            out[ov_start:ov_end, :] -= ch[:fade_steps, :] * (1.0 - fade_in)
            out[ov_start:ov_end, :] += ch[:fade_steps, :] * fade_in
            # 重みも同様に滑らかに（単純化のため 1 のままでも大抵大丈夫）
    # 平均化（重み0回避）
    weight[weight == 0] = 1.0
    out /= weight
    return out
