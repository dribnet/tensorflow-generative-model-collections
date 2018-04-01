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

        # launch the graph in a session
        self.model.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.model.load(filename)
        print(" [*] Loading finished!")

        # self.invert_models = def_invert_models(self.net, layer='conv4', alpha=0.002)

    def encode_images(self, images, cond=None):
        # print("images: ", images.shape, images[0][0])
        rec, zs, _  = invert_images_opt(self.invert_models, images)

        # pixels = (255 * images).astype(np.uint8)
        # pixels = np.swapaxes(pixels,1,2)
        # pixels = np.swapaxes(pixels,2,3)
        # # print("SHAPE: {} {}".format(pixels.shape, pixels.dtype))
        # _, zs, _  = invert_images_CNN_opt(self.invert_models, pixels, solver='cnn_opt', npx=self.model_G.npx)
        # print(zs)
        # print("Zs SHAPE: {}".format(zs.shape))
        return zs

    def get_zdim(self):
        # ?
        return self.model.z_dim

    def sample_at(self, z):
        # tot_num_samples = min(self.model.sample_num, self.model.batch_size)
        # image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        self.model.batch_size = len(z)
        samples = self.session.run(self.model.fake_images, feed_dict={self.model.z: z})
        channel_first = np.rollaxis(samples, 3, 1)

        return channel_first
        # save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #             'outputs/test/sample_01.png')

        # mapped_latents = z.astype(np.float32)
        # samples = self.net.gen_fn(mapped_latents, mapped_latents)
        # samples = np.clip(samples, 0, 1)

        # self.net.example_latents = z.astype(np.float32)
        # self.net.example_labels = self.net.example_latents
        # self.net.latents_var = T.TensorType('float32', [False] * len(self.net.example_latents.shape))('latents_var')
        # self.net.labels_var  = T.TensorType('float32', [False] * len(self.net.example_latents.shape)) ('labels_var')

        # self.net.images_expr = self.net.G.eval(self.net.example_latents, self.net.labels_var, ignore_unused_inputs=True)
        # self.net.images_expr = misc.adjust_dynamic_range(self.net.images_expr, [-1,1], [0,1])

        # if not self.have_compiled:
        #     train.imgapi_compile_gen_fn(self.net)
        #     self.have_compiled = True

        # samples = self.net.gen_fn(self.net.example_latents, self.net.example_labels)
        # samples = np.clip(samples, 0, 1)

        # print("Samples: ", samples.shape)
        # samples = (samples + 1.0) / 2.0
        # return samples
