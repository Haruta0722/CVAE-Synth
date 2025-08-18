import numpy as np
import librosa
import tensorflow as tf
import os


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

    # バッチ化
    cond_vector = np.expand_dims(cond_vector, axis=0)  # (1,27)

    # 潜在ベクトルをランダムにサンプリング
    z_sample = tf.random.normal(
        shape=(1, cvae_model.encoder.output[0].shape[-1])
    )

    # Decoderでスペクトログラム生成
    generated_spec = cvae_model.decoder([z_sample, cond_vector])
    generated_spec = tf.squeeze(
        generated_spec, axis=0
    ).numpy()  # (n_mels, frames, 1)
    generated_spec = np.squeeze(generated_spec, axis=-1)  # (n_mels, frames)

    # Mel→Linearスペクトログラム変換（librosaで近似）
    # まずは振幅スペクトログラムにスケール
    mel_basis = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=n_mels)
    inv_mel = np.linalg.pinv(mel_basis)
    linear_spec = np.maximum(
        0, np.dot(inv_mel, generated_spec)
    )  # 非負にクリッピング

    # Griffin-Limで波形復元
    waveform = librosa.griffinlim(
        linear_spec, n_iter=n_iter, hop_length=hop_length, win_length=1024
    )

    return waveform


def make_condition_vector(
    pitch,
    waveform_type,
    thickness=0.5,
    brightness=0.5,
    distortion=0.0,
    delay_reverb=0.0,
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
    cont_vec = tf.convert_to_tensor(
        [
            thickness,
            brightness,
            distortion,
            delay_reverb,
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

    cond_vector = tf.concat([pitch_vec, wave_vec, cont_vec], axis=0)
    return cond_vector.numpy()


def process_preset_audio(
    audio_path, preset_params, sr=44100, n_fft=1024, hop_length=256, n_mels=128
):
    """
    audio_path: 音声ファイルパス
    preset_params: 27次元の条件ラベル（dict形式 or list）
    """
    # --- 音声読み込み ---
    y, _ = librosa.load(audio_path, sr=sr)

    # 2秒固定にする場合（切り取り or パディング）
    target_len = sr * 2
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # --- Melスペクトログラム ---
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)
    S_mel = librosa.feature.melspectrogram(S=S_mag, sr=sr, n_mels=n_mels)
    S_mel = S_mel / np.max(S_mel)  # 0-1正規化

    # --- 条件ベクトル作成 ---
    # preset_paramsはdictの場合
    cond_vector = []
    # 音高 12次元 one-hot
    pitch = preset_params["pitch"]  # 0〜11
    pitch_vec = np.zeros(12)
    pitch_vec[pitch] = 1
    cond_vector.extend(pitch_vec)

    # 波形パターン 4次元 one-hot
    wave = preset_params["waveform"]  # 0〜3
    wave_vec = np.zeros(4)
    wave_vec[wave] = 1
    cond_vector.extend(wave_vec)

    # 残り 11次元 連続値
    for key in [
        "thickness",
        "brightness",
        "distortion",
        "delay_reverb",
        "lowpass",
        "highpass",
        "sweep_close",
        "sweep_open",
        "attack",
        "release",
        "side",
    ]:
        cond_vector.append(preset_params.get(key, 0.5))  # デフォルト0.5

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
