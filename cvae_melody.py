import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


# ===== 設定 =====
n_mels = 128  # メルスペクトログラムの周波数ビン数
frames = 173  # STFTフレーム数 (例: 2秒 @ 44.1kHz, hop_length=256)
cond_dim = 27  # 条件ベクトル次元
latent_dim = 64  # 潜在次元


# ===== Encoder =====
def build_encoder(input_shape=(n_mels, frames, 1), cond_dim=27, latent_dim=64):
    spec_in = layers.Input(shape=input_shape, name="spectrogram")
    cond_in = layers.Input(shape=(cond_dim,), name="condition")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(spec_in)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # 条件ベクトル結合
    x = layers.Concatenate()([x, cond_in])
    x = layers.Dense(128, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    encoder = Model([spec_in, cond_in], [z_mean, z_log_var], name="encoder")
    return encoder


# ===== Sampling layer =====
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ===== Decoder =====
def build_decoder(output_shape=(n_mels, frames, 1), cond_dim=27, latent_dim=64):
    z_in = layers.Input(shape=(latent_dim,), name="z")
    cond_in = layers.Input(shape=(cond_dim,), name="condition")

    x = layers.Concatenate()([z_in, cond_in])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(np.prod(output_shape), activation="sigmoid")(x)
    x = layers.Reshape(output_shape)(x)

    decoder = Model([z_in, cond_in], x, name="decoder")
    return decoder


optimizer = tf.keras.optimizers.Adam(1e-4)
mse_loss = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(model, x, cond):
    with tf.GradientTape() as tape:
        # Encoder
        z_mean, z_log_var = model.encoder([x, cond])
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps

        # Decoder
        x_recon = model.decoder([z, cond])

        # Loss
        recon_loss = mse_loss(x, x_recon)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        total_loss = recon_loss + kl_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss


# ===== CVAEモデル =====
class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, inputs):
        x, cond = inputs
        z_mean, z_log_var = self.encoder([x, cond])
        z = self.sampling([z_mean, z_log_var])
        x_recon = self.decoder([z, cond])
        # VAE loss計算
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_recon))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(reconstruction_loss + kl_loss)
        return x_recon
