# inference.py
import numpy as np
import soundfile as sf
from cvae import CVAE
from tensorflow.keras import mixed_precision
from cvae_utils import wav_to_mel, mel_to_wav

mixed_precision.set_global_policy("mixed_float16")

LATENT_DIM = 64
COND_DIM = 64
MEL_BINS = 128
TIME_STEPS = 128
SR = 22050


# 条件ベクトル例（正規化済み）
cond = np.array(
    [[0.5, 1.0, 0.3, 0.7, 0.4, 0.2, 0.1, 0.8, 0.0]], dtype="float32"
)

# モデルをロード（学習済みの重みファイル指定）
input_shape = (TIME_STEPS, MEL_BINS, 1)
cvae = CVAE(input_shape, COND_DIM)
cvae.load_weights("checkpoints/cvae_weights.h5")

mel_mono = wav_to_mel("mono_input.wav")
diff_mel, _, _ = cvae([mel_mono, cond])
mel_unison = mel_mono + diff_mel.numpy()

wav_out = mel_to_wav(mel_unison)
sf.write("generated_unison.wav", wav_out, SR)
