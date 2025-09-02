# train_cvae.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

LATENT_DIM = 64
COND_DIM = 64
MEL_BINS = 128
TIME_STEPS = 128  # STFT長さに依存


def build_encoder(input_shape, cond_dim):
    x_in = layers.Input(shape=input_shape)
    cond_in = layers.Input(shape=(cond_dim,))

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x_in)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)

    # 条件を混ぜる
    c = layers.Dense(128, activation="relu")(cond_in)
    xc = layers.Concatenate()([x, c])

    mu = layers.Dense(LATENT_DIM)(xc)
    logvar = layers.Dense(LATENT_DIM)(xc)

    return Model([x_in, cond_in], [mu, logvar], name="encoder")


def build_decoder(output_shape, cond_dim):
    z_in = layers.Input(shape=(LATENT_DIM,))
    cond_in = layers.Input(shape=(cond_dim,))

    # FiLMで条件を注入
    c = layers.Dense(128, activation="relu")(cond_in)
    merged = layers.Concatenate()([z_in, c])

    x = layers.Dense(8 * 8 * 128, activation="relu")(merged)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding="same", activation="relu"
    )(x)
    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding="same", activation="relu"
    )(x)
    out = layers.Conv2DTranspose(1, 3, strides=2, padding="same")(x)

    return Model([z_in, cond_in], out, name="decoder")


class CVAE(Model):
    def __init__(self, input_shape, cond_dim):
        super().__init__()
        self.encoder = build_encoder(input_shape, cond_dim)
        self.decoder = build_decoder(input_shape, cond_dim)

    def sample_latent(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    def call(self, inputs):
        x, cond = inputs
        mu, logvar = self.encoder([x, cond])
        z = self.sample_latent(mu, logvar)
        recon = self.decoder([z, cond])
        return recon, mu, logvar


# === 学習ループ（ダミー例） ===
input_shape = (TIME_STEPS, MEL_BINS, 1)
cvae = CVAE(input_shape, COND_DIM)

optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(x, cond, y):
    with tf.GradientTape() as tape:
        recon, mu, logvar = cvae([x, cond])
        recon_loss = tf.reduce_mean(tf.abs(y - recon))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + logvar - tf.square(mu) - tf.exp(logvar)
        )
        loss = recon_loss + 0.01 * kl_loss
    grads = tape.gradient(loss, cvae.trainable_variables)
    optimizer.apply_gradients(zip(grads, cvae.trainable_variables))
    return loss


# ダミーデータでテスト
x_dummy = np.random.randn(4, TIME_STEPS, MEL_BINS, 1).astype("float32")
c_dummy = np.random.randn(4, COND_DIM).astype("float32")
y_dummy = np.random.randn(4, TIME_STEPS, MEL_BINS, 1).astype("float32")

loss = train_step(x_dummy, c_dummy, y_dummy)
print("Loss:", loss.numpy())
