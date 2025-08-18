import tensorflow as tf
from cvae_melody_utils import make_condition_vector, generate_waveform

# 学習済みモデル読み込み
cvae = tf.keras.models.load_model("cvae_model", compile=False)

# 条件ベクトルを作成
cond_vec = make_condition_vector(
    pitch=0, waveform_type=1, thickness=0.7, brightness=0.9
)

# Mel スペクトログラム生成
mel_pred = cvae.sample(y=tf.expand_dims(cond_vec, axis=0))

# Griffin-Lim や HiFi-GAN で波形復元
waveform = generate_waveform(mel_pred)
