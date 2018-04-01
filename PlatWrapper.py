import tensorflow as tf
from VAE import VAE
import numpy as np

class VAEp:
    def __init__(self, filename=None, model=None):
        if model is not None:
            # error
            return

        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.last_batch_size = 64
        self.model = VAE(self.session,
                           epoch=20,
                           batch_size=self.last_batch_size,
                           z_dim=20,
                           dataset_name="mnist",
                           checkpoint_dir=filename,
                           result_dir="results",
                           log_dir="logs")

        # build graph
        self.model.build_model()

        """ Loss Function """
        # encoding
        # mu, sigma = self.model.encoder(self.inputs, is_training=False, reuse=True)

        # sampling by re-parameterization technique
        # self.z_fn = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # launch the graph in a session
        self.model.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.model.load(filename)
        print(" [*] Loading finished!")

        # self.invert_models = def_invert_models(self.net, layer='conv4', alpha=0.002)

    def encode_images(self, images, cond=None):
        channel_last = np.rollaxis(images, 1, 4)
        z = self.session.run(self.model.mu, feed_dict={self.model.inputs: channel_last})
        return z

    def get_zdim(self):
        return self.model.z_dim

    def sample_at(self, z):
        samples = self.session.run(self.model.fake_images, feed_dict={self.model.z: z})
        channel_first = np.rollaxis(samples, 3, 1)
        return channel_first
