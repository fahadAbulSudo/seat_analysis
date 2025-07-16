import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Config ---
img_height, img_width = 768, 768  # ⬅️ higher resolution for better quality
channels = 3
batch_size = 2
epochs = 50
latent_dim = 32
distribution = "uniform"
learning_rate = 1e-4  # From Optuna

# --- Data Loader ---
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_data = datagen.flow_from_directory(
    '/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/Inspector Pictures from Google Notes/dataset',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    color_mode="rgb"
)

# --- Encoder ---
def get_encoder(latent_dim):
    encoder_inputs = layers.Input(shape=(img_height, img_width, channels))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation='relu')(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    return encoder_inputs, z_mean, z_log_var

# --- Decoder ---
def get_decoder(latent_dim):
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense((img_height // 16) * (img_width // 16) * 256, activation='relu')(decoder_inputs)
    x = layers.Reshape((img_height // 16, img_width // 16, 256))(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation='relu')(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation='relu')(x)
    decoder_outputs = layers.Conv2DTranspose(channels, 3, activation="sigmoid", padding="same")(x)
    return decoder_inputs, decoder_outputs

# --- Sampling ---
def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.uniform(shape=(tf.shape(z_mean)[0], latent_dim), minval=-1, maxval=1)
    return z_mean + tf.exp(0.5 * z_log_var) * eps

# --- VAE Class ---
class VAE(Model):
    def __init__(self, encoder, decoder, sampling_fn, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling_fn = sampling_fn

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_clipped = tf.clip_by_value(reconstruction, 1e-7, 1 - 1e-7)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.backend.binary_crossentropy(data, reconstruction_clipped), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# --- Build Model ---
encoder_inputs, z_mean, z_log_var = get_encoder(latent_dim)
z = layers.Lambda(sampling)([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
decoder_inputs, decoder_outputs = get_decoder(latent_dim)
decoder = Model(decoder_inputs, decoder_outputs, name="decoder")
vae = VAE(encoder, decoder, sampling)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# --- Train ---
vae.fit(train_data, epochs=epochs, verbose=1)

# --- Save Models ---
vae.save('trained_vae_model.h5')
encoder.save('vae_encoder.h5')
decoder.save('vae_decoder.h5')
