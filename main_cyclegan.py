# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import numpy as np
import os
import tensorflow as tf
from models.gan_ops import *
from datetime import datetime
from tensorflow.contrib import summary

now = datetime.now()

INPUT_DIM = 128
OUTPUT_DIM = 3

date = now.strftime("%Y%m%d-%H%M%S")


def generator_fn(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        dtype='float32',  # Data type to use for activations and outputs.
        normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
        latent_size=INPUT_DIM,
        scope='Generator',
        mapping_lrmul=0.01,  # Learning rate multiplier for the mapping layers.
        use_wscale=True,  # Enable equalized learning rate?
        mapping_nonlinearity='lrelu',
        **_kwargs):  # Ignore unrecognized keyword args.

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[mapping_nonlinearity]
    latents_in.set_shape([None, latent_size])
    latents_in = tf.cast(latents_in, dtype)
    x = latents_in

    # Normalize latents.
    if normalize_latents:
        x = pixel_norm(x)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('Dense%d' % 1):
            x = dense(x, fmaps=1024, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)
        with tf.variable_scope('Dense%d' % 2):
            x = dense(x, fmaps=7 * 7 * 128, gain=gain)
            x = tf.reshape(x, [-1, 128, 7, 7])
            x = act(instance_norm(x))
        with tf.variable_scope('Dense%d' % 3):
            x = upscale2d_conv2d(x, 64, kernel=3, gain=gain, use_wscale=use_wscale)
            x = act(instance_norm(x))
        with tf.variable_scope('Dense%d' % 4):
            x = upscale2d_conv2d(x, 1, kernel=3, gain=gain, use_wscale=use_wscale)
            x = tf.tanh(x)
            return x


def discriminator_fn(
        images_in,  # First input: Images [minibatch, channel, height, width].
        labels_in,  # Second input: Labels [minibatch, label_size].
        num_channels=1,  # Number of input color channels. Overridden based on dataset.
        resolution=28,  # Input resolution. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        fmap_base=16,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu',
        use_wscale=True,  # Enable equalized learning rate?
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        fused_scale='auto',  # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        blur_filter=[1, 2, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        **_kwargs):  # Ignore unrecognized keyword args.
    resolution_log2 = int(np.log2(resolution))

    # assert resolution == 2 ** resolution_log2 and resolution >= 4

    def nf(stage):
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    def blur(x):
        return blur2d(x, blur_filter) if blur_filter else x

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    scores_out = None

    # Building blocks.
    def fromrgb(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res - 1), kernel=1, gain=gain, use_wscale=use_wscale)))

    def block(x, res):  # res = 2..resolution_log2

        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            if res >= 3:  # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res - 1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Conv1_down'):
                    x = act(apply_bias(
                            conv2d_downscale2d(blur(x), fmaps=nf(res - 2), kernel=3, gain=gain, use_wscale=use_wscale,
                                               fused_scale=fused_scale)))
            else:  # 4x4
                x = tf.transpose(x, [0, 2, 3, 1])
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res - 1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res - 2), gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=max(label_size, 1), gain=1, use_wscale=use_wscale))
            return x

    # Fixed structure: simple and efficient, but does not support progressive growing.
    x = fromrgb(images_in, resolution_log2)
    for res in range(resolution_log2, 2, -1):
        x = block(x, res)

    scores_out = block(x, 2)

    # Label conditioning from "Which Training Methods for GANs do actually Converge?"
    if label_size:
        with tf.variable_scope('LabelSwitch'):
            scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out


def gen_model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()

    generator_inputs = labels
    real_data = features

    with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
        gan_model = tf.contrib.gan.gan_model(generator_fn, discriminator_fn, real_data, generator_inputs)

    predictions = gan_model.generated_data
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        gan_loss = tf.contrib.gan.gan_loss(gan_model, add_summaries=False)
        loss = gan_loss.generator_loss

        # define train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        dummy_optimizer = tf.train.AdamOptimizer(learning_rate=0.005)

        # wrapper to make the optimizer work with TPUs
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        gan_train_ops = tf.contrib.gan.gan_train_ops(gan_model, gan_loss, optimizer, dummy_optimizer)

        # tf.contrib.gan's train op does not manage global steps in it

        train_op = tf.group(gan_train_ops.generator_train_op, global_step.assign_add(1))

    def host_call_fn(gs, loss, img_grid):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed hparams.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.a
        model_dir = args.model_dir + '/' + date + "/"

        print(model_dir)

        with summary.create_file_writer(
                model_dir, max_queue=10).as_default():
            with summary.always_record_summaries():
                summary.scalar('loss_g', loss[0], step=gs)
                summary.image('img_grid', img_grid, step=gs)

                return summary.all_summary_ops()

    # To log the loss, current learning rate, and epoch for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    gs_t = tf.reshape(global_step, [1])
    loss_t = tf.reshape(gan_loss.generator_loss, [1])

    feat = tf.transpose(features, perm=[0, 2, 3, 1])
    img_grid = tf.contrib.gan.eval.image_grid(
            feat,
            [4, 8],
            image_shape=(28, 28),
            num_channels=1
            )

    host_call = (host_call_fn, [gs_t, loss_t, img_grid])

    # TPU version of EstimatorSpec
    return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            host_call=host_call,
            train_op=train_op)


def dis_model_fn(features, labels, mode, params):
    # build model
    global_step = tf.train.get_global_step()

    generator_inputs = labels
    real_data = features

    with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
        gan_model = tf.contrib.gan.gan_model(generator_fn, discriminator_fn, real_data, generator_inputs)

    predictions = {
            'discriminator_gen_outputs':  gan_model.discriminator_gen_outputs,
            'discriminator_real_outputs': gan_model.discriminator_real_outputs}
    loss = None
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        # define loss
        gan_loss = tf.contrib.gan.gan_loss(gan_model, add_summaries=False)
        loss = gan_loss.discriminator_loss

        # define train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        dummy_optimizer = tf.train.AdamOptimizer(learning_rate=0.005)

        # wrapper to make the optimizer work with TPUs
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        gan_train_ops = tf.contrib.gan.gan_train_ops(gan_model, gan_loss, dummy_optimizer, optimizer)

        # tf.contrib.gan's train op does not manage global steps in it
        train_op = tf.group(gan_train_ops.discriminator_train_op, global_step.assign_add(1))

    # TPU version of EstimatorSpec
    return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def parsing(serialized_example):
    """Parses a single Example into image and label tensors."""
    features = tf.parse_single_example(
            serialized_example,
            features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label':     tf.FixedLenFeature([], tf.int64)  # label is unused
                    })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([28 * 28])
    image = tf.reshape(image, [1, 28, 28])

    # Normalize the values of the image from [0, 255] to [-1.0, 1.0]
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0

    label = tf.cast(tf.reshape(features['label'], shape=[]), dtype=tf.int32)
    return image, label


def train_input_fn(mode, params):
    # make some fake noise

    data_file = 'gs://cloud-tpu-epfl-stockage/mnist/data/train.tfrecords'
    dataset = tf.data.TFRecordDataset(data_file)
    dataset = dataset.map(parsing).cache()
    dataset = dataset.shuffle(1024)
    dataset = dataset.prefetch(args.train_batch_size)
    dataset = dataset.batch(args.train_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()
    random_noise = tf.random_normal([args.train_batch_size, 128])

    return images, random_noise


def main(args):
    # pass the args as params so the model_fn can use
    # the TPU specific args
    args, _ = parser.parse_known_args()

    params = vars(args)
    tpu = 'chocoarthur'
    tpu_zone = 'us-central1-f'
    gcp_project = 'cloud-tpu-epfl'
    model_dir = args.model_dir + '/' + date + "/"
    num_shards = 8
    iterations_per_loop = 8000
    # additional configs required for using TPUs
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu,
            zone=tpu_zone,
            project=gcp_project)

    tpu_config = tf.contrib.tpu.TPUConfig(
            num_shards=num_shards,
            per_host_input_for_training=True,
            iterations_per_loop=iterations_per_loop)

    # use the TPU version of RunConfig
    gen_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=os.path.join(model_dir, 'generator'),
            tpu_config=tpu_config,
            save_checkpoints_steps=None,
            save_summary_steps=100)

    dis_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=os.path.join(model_dir, 'discriminator'),
            tpu_config=tpu_config,
            save_checkpoints_steps=None,
            save_summary_steps=100)

    # TPUEstimator
    gen_estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=True,
            model_fn=gen_model_fn,
            config=gen_config,
            params=params,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.train_batch_size,
            export_to_tpu=False)

    dis_estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=True,
            model_fn=dis_model_fn,
            config=dis_config,
            params=params,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.train_batch_size,
            export_to_tpu=False)

    # manage the training loop
    for _ in range(10):
        print('Training Discriminator')
        dis_estimator.train(train_input_fn, steps=100)
        print('Training Generator')
        gen_estimator.train(train_input_fn, steps=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--model-dir',
            type=str,
            default='gs://cloud-tpu-epfl-stockage/cyclegan/',
            help='Location to write checkpoints and summaries to.  Must be a GCS URI when using Cloud TPU.')
    parser.add_argument(
            '--train-batch-size',
            type=int,
            default=256,
            help='The training batch size.  The training batch is divided evenly across the TPU cores.')
    parser.add_argument(
            '--save-checkpoints-steps',
            type=int,
            default=0,
            help='The number of training steps before saving each checkpoint.')
    parser.add_argument(
            '--use-tpu',
            default=True,
            action='store_true',
            help='Whether to use TPU.')
    parser.add_argument(
            '--tpu',
            default='chocoarthur',
            help='The name or GRPC URL of the TPU node.  Leave it as `None` when training on CMLE.')
    tf.reset_default_graph()
    args = parser.parse_args()
    main(args)
