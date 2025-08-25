import numpy as np
import librosa
import tensorflow as tf
import os
import soundfile as sf


def ms_to_lr(mid_wave, side_wave):
    """Mid/Side 波形を Left/Right に変換"""
    left = (mid_wave + side_wave) / np.sqrt(2)
    right = (mid_wave - side_wave) / np.sqrt(2)
    return np.stack([left, right], axis=-1)  # shape: (samples, 2)


def generate_waveform(
    cvae_model,
    cond_vector,
    n_mels=128,
    frames=173,
    n_iter=60,
    hop_length=256,
    sr=44100,
):
    """
    cvae_model: 学習済みCVAEモデル
    cond_vector: 27次元条件ベクトル
    n_iter: Griffin-Lim反復回数
    """

    # --- 条件ベクトル準備 ---
    cond_vector = np.expand_dims(cond_vector, axis=0)  # (1, cond_dim)
    cond_vector = tf.convert_to_tensor(cond_vector, dtype=tf.float32)
    # --- 潜在ベクトルサンプリング ---
    z_sample = tf.random.normal(
        shape=(1, cvae_model.encoder.output[0].shape[-1])
    )

    # --- デコーダで M/S メルスペクトログラム生成 ---
    generated_spec = cvae_model.decoder([z_sample, cond_vector])
    # shape: (1, n_mels, frames, 2)
    generated_spec = tf.squeeze(
        generated_spec, axis=0
    ).numpy()  # (n_mels, frames, 2)

    # --- Melフィルタ逆変換行列 ---
    mel_basis = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=n_mels)
    inv_mel = np.linalg.pinv(mel_basis)

    # --- Mid / Side の各チャネルを波形に復元 ---
    mid_spec = generated_spec[..., 0]  # (n_mels, frames)
    side_spec = generated_spec[..., 1]

    def mel_to_wave(mel_spec):
        # Mel → Linear
        linear_spec = np.maximum(0, np.dot(inv_mel, mel_spec))
        # Griffin-Limで波形復元
        return librosa.griffinlim(
            linear_spec, n_iter=n_iter, hop_length=hop_length, win_length=1024
        )

    mid_wave = mel_to_wave(mid_spec)
    side_wave = mel_to_wave(side_spec)

    # --- M/S → L/R に変換 ---
    stereo_wave = ms_to_lr(mid_wave, side_wave)  # shape: (samples, 2)
    # 保存
    sf.write("generated.wav", stereo_wave, 44100)
    return stereo_wave


def make_condition_vector(
    pitch,
    waveform_type,
    layer_waveform,
    thickness=0.5,
    brightness=0.5,
    distortion=0.0,
    delay_reverb=0.0,
    texture=0.0,
    lowpass=1.0,
    highpass=0.0,
    sweep_close=0.0,
    sweep_open=0.0,
    attack=0.5,
    release=0.5,
    side=0.5,
):

    pitch_vec = tf.one_hot(pitch, 12)
    wave_vec = tf.one_hot(waveform_type, 4)
    layer_vec = tf.one_hot(layer_waveform, 4)
    cont_vec = tf.convert_to_tensor(
        [
            thickness,
            brightness,
            distortion,
            delay_reverb,
            texture,
            lowpass,
            highpass,
            sweep_close,
            sweep_open,
            attack,
            release,
            side,
        ],
        dtype=tf.float32,
    )

    cond_vector = tf.concat([pitch_vec, wave_vec, layer_vec, cont_vec], axis=0)
    return cond_vector.numpy()


def process_preset_audio(
    audio_path, preset_params, sr=44100, n_fft=1024, hop_length=256, n_mels=128
):
    """
    audio_path: 音声ファイルパス
    preset_params: 27次元の条件ラベル（dict形式 or list）
    """
    # --- ステレオ読み込み ---
    y, _ = librosa.load(audio_path, sr=sr, mono=False)

    # (2, n_samples) → (L, R)
    if y.ndim == 1:  # モノラルしかなかった場合はコピー
        y = np.stack([y, y], axis=0)

    L, R = y[0], y[1]

    # --- Mid/Side変換 ---
    M = (L + R) / np.sqrt(2)
    S = (L - R) / np.sqrt(2)

    # 2秒固定
    target_len = sr * 2

    def fix_length(sig):
        if len(sig) < target_len:
            return np.pad(sig, (0, target_len - len(sig)))
        else:
            return sig[:target_len]

    M = fix_length(M)
    S = fix_length(S)

    # --- STFT ---
    S_M = librosa.stft(M, n_fft=n_fft, hop_length=hop_length)
    S_S = librosa.stft(S, n_fft=n_fft, hop_length=hop_length)

    # --- メルスペクトログラム ---
    S_mag_M = np.abs(S_M)
    S_mag_S = np.abs(S_S)

    S_mel_M = librosa.feature.melspectrogram(S=S_mag_M, sr=sr, n_mels=n_mels)
    S_mel_S = librosa.feature.melspectrogram(S=S_mag_S, sr=sr, n_mels=n_mels)

    # 正規化（それぞれ独立 or 全体のmaxで統一するかは設計次第）
    S_mel_M = S_mel_M / np.max(S_mel_M)
    S_mel_S = S_mel_S / np.max(S_mel_S)

    # --- チャネル結合 (M,S) ---
    S_mel = np.stack([S_mel_M, S_mel_S], axis=-1)  # (n_mels, frames, 2)

    # --- 条件ベクトル作成 ---
    cond_vector = []
    pitch = preset_params["pitch"]
    pitch_vec = np.zeros(12)
    pitch_vec[pitch] = 1
    cond_vector.extend(pitch_vec)

    wave = preset_params["waveform"]
    wave_vec = np.zeros(4)
    wave_vec[wave] = 1
    cond_vector.extend(wave_vec)

    wave = preset_params["layer_waveform"]
    wave_vec = np.zeros(4)
    wave_vec[wave] = 1
    cond_vector.extend(wave_vec)

    for key in [
        "thickness",
        "brightness",
        "distortion",
        "delay_reverb",
        "texture",
        "lowpass",
        "highpass",
        "sweep_close",
        "sweep_open",
        "attack",
        "release",
        "side",
    ]:
        cond_vector.append(preset_params.get(key, 0.5))

    cond_vector = np.array(cond_vector, dtype=np.float32)  # (27,)

    return S_mel, cond_vector


def create_dataset(
    audio_dir, preset_json, sr=44100, n_fft=1024, hop_length=256, n_mels=128
):
    """
    audio_dir: 音声ファイル格納フォルダ
    preset_json: { "filename.wav": {preset_params} } のJSONファイル
    """
    X_specs = []
    X_cond = []

    for fname, params in preset_json.items():
        path = os.path.join(audio_dir, fname)
        if not os.path.exists(path):
            continue
        S_mel, cond_vec = process_preset_audio(
            path,
            params,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        X_specs.append(S_mel[..., np.newaxis])  # チャネル次元追加
        X_cond.append(cond_vec)

    X_specs = np.array(X_specs, dtype=np.float32)
    X_cond = np.array(X_cond, dtype=np.float32)
    return X_specs, X_cond
