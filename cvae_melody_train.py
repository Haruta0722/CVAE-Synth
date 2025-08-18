import json
from cvae_melody_utils import (
    create_dataset,
    generate_waveform,
    make_condition_vector,
)
from cvae_melody import CVAE, build_encoder, build_decoder, train_step
import tensorflow as tf

audio_dir = "./preset_wav"
with open("./preset.json") as f:
    preset_json = json.load(f)

X_spec, X_cond = create_dataset(audio_dir, preset_json)

print("X_spec:", X_spec.shape)  # (N, n_mels, frames, 1)
print("X_cond:", X_cond.shape)  # (N, 27)

batch_size = 16
dataset = tf.data.Dataset.from_tensor_slices((X_spec, X_cond))
dataset = dataset.shuffle(buffer_size=100).batch(batch_size)

# ===== モデル構築 =====
encoder = build_encoder()
decoder = build_decoder()
cvae = CVAE(encoder, decoder)


epochs = 50

for epoch in range(epochs):
    total_loss = 0
    for step, (x_batch, cond_batch) in enumerate(dataset):
        loss = train_step(cvae, x_batch, cond_batch)
        total_loss += loss.numpy()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.4f}")

cvae.save("cvae_model")

# ===== 学習例 =====
# X_spec: (N, n_mels, frames, 1)
# X_cond: (N, 27)
# cvae.fit([X_spec, X_cond], X_spec, batch_size=16, epochs=50)

# 条件ラベル例
cond_vec = make_condition_vector(
    pitch=0, waveform_type=1, thickness=0.7, brightness=0.9
)
waveform = generate_waveform(cvae, cond_vec)  # 先ほどのGriffin-Lim関数
