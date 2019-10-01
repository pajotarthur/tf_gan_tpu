# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Train a ResNet-50 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from common import tpu_profiler_hook
from input import imagenet_input
from models import resnet_model
from tensorflow.contrib import summary
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator
from tensorflow.contrib.training.python.training.hparam import HParams


tf.logging.set_verbosity(tf.logging.INFO)

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'



# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


hparams = HParams(

    steps_per_eval=1251,
    iterations_per_loop=1251,
    skip_host_call=False,
    eval_timeout=None,
    use_tpu=True,
    tpu='chocoarthur',
    model_dir='gs://cloud-tpu-epfl-stockage/resnet',

    image_size=128,
    train_batch_size= 1024,
    eval_batch_size = 1024,
    num_train_images=11259,
    num_eval_images=2000,
    num_label_classes=1000,
    train_steps=200,

    resnet_depth=50,

    data_dir='gs://cloud-tpu-epfl-stockage/imagenet',
    log_step_count_steps=100,

    base_learning_rate=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    label_smoothing=0.0,
    poly_rate=0,

    mode='train_and_eval',

    data_format='channels_last',
    transpose_input=True,
    use_async_checkpointing=True,
    use_cache=True,
    precision='float32',
    export_to_tpu=False,
    export_dir=None,
    num_cores=8,
    num_parallel_calls=8,
    profile_every_n_steps=1000,

    dropblock_size=7,
    dropblock_keep_prob=0.9,
    dropblock_groups='',

)

print(hparams)

def get_lr_schedule(train_steps, num_train_images, train_batch_size):
    """learning rate schedule."""
    steps_per_epoch = np.floor(num_train_images / train_batch_size)
    train_epochs = train_steps / steps_per_epoch
    return [  # (multiplier, epoch to start) tuples
        (1.0, np.floor(5 / 90 * train_epochs)),
        (0.1, np.floor(30 / 90 * train_epochs)),
        (0.01, np.floor(60 / 90 * train_epochs)),
        (0.001, np.floor(80 / 90 * train_epochs))
    ]


def learning_rate_schedule(train_steps, current_epoch):
    """Handles linear scaling rule, gradual warmup, and LR decay.

    The learning rate starts at 0, then it increases linearly per step.
    After 5 epochs we reach the base learning rate (scaled to account
      for batch size).
    After 30, 60 and 80 epochs the learning rate is divided by 10.
    After 90 epochs training stops and the LR is set to 0. This ensures
      that we train for exactly 90 epochs for reproducibility.

    Args:
      train_steps: `int` number of training steps.
      current_epoch: `Tensor` for current epoch.

    Returns:
      A scaled `Tensor` for current learning rate.
    """
    scaled_lr = hparams.base_learning_rate * (hparams.train_batch_size / 256.0)

    lr_schedule = get_lr_schedule(
        train_steps=train_steps,
        num_train_images=hparams.num_train_images,
        train_batch_size=hparams.train_batch_size)
    decay_rate = (scaled_lr * lr_schedule[0][0] *
                  current_epoch / lr_schedule[0][1])
    for mult, start_epoch in lr_schedule:
        decay_rate = tf.where(current_epoch < start_epoch,
                              decay_rate, scaled_lr * mult)
    return decay_rate


def resnet_model_fn(features, labels, mode, params):
    """The model_fn for ResNet to be used with TPUEstimator.

    Args:
      features: `Tensor` of batched images. If transpose_input is enabled, it
          is transposed to device layout and reshaped to 1D tensor.
      labels: `Tensor` of labels for the data samples
      mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
      params: `dict` of parameters passed to the model from the TPUEstimator,
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `TPUEstimatorSpec` for the model
    """
    print(features.shape)

    if isinstance(features, dict):
        features = features['features']


    # In most cases, the default data format NCHW instead of NHWC should be
    # used for a significant performance boost on GPU/TPU. NHWC should be used
    # only if the network needs to be run on CPU since the pooling operations
    # are only supported on NHWC.
    if hparams.data_format == 'channels_first':
        assert not hparams.transpose_input  # channels_first only for GPU
        features = tf.transpose(features, [0, 3, 1, 2])

    # print(features.shape)

    # tf.Print(features, [tf.shape(features)], 'coucou')
    if hparams.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
        image_size = tf.constant(128) # tf.sqrt(tf.shape(features)[0] / (3 * tf.shape(labels)[0]))
        # print(features.shape)
        # print(image_size)
        # print(tf.shape(features)[0])
        # print(3 * tf.shape(labels)[0])

        # print(image_size, image_size, 3)
        features = tf.reshape(features, [image_size, image_size, 3, -1])
        print(features.shape)

        features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC
        print(features.shape)

    # Normalize the image to zero mean and unit variance.
    features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
    features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

    # DropBlock keep_prob for the 4 block groups of ResNet architecture.
    # None means applying no DropBlock at the corresponding block group.
    dropblock_keep_probs = [None] * 4
    if hparams.dropblock_groups:
        # Scheduled keep_prob for DropBlock.
        train_steps = tf.cast(hparams.train_steps, tf.float32)
        current_step = tf.cast(tf.train.get_global_step(), tf.float32)
        current_ratio = current_step / train_steps
        dropblock_keep_prob = (1 - current_ratio * (1 - hparams.dropblock_keep_prob))

        # Computes DropBlock keep_prob for different block groups of ResNet.
        dropblock_groups = [int(x) for x in hparams.dropblock_groups.split(',')]
        for block_group in dropblock_groups:
            if block_group < 1 or block_group > 4:
                raise ValueError(
                    'dropblock_groups should be a comma separated list of integers '
                    'between 1 and 4 (dropblcok_groups: {}).'
                        .format(hparams.dropblock_groups))
            dropblock_keep_probs[block_group - 1] = 1 - (
                    (1 - dropblock_keep_prob) / 4.0 ** (4 - block_group))

    # This nested function allows us to avoid duplicating the logic which
    # builds the network, for different values of --precision.
    def build_network():
        network = resnet_model.resnet_v1(
            resnet_depth=hparams.resnet_depth,
            num_classes=hparams.num_label_classes,
            dropblock_size=hparams.dropblock_size,
            dropblock_keep_probs=dropblock_keep_probs,
            data_format=hparams.data_format)
        return network(
            inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))


    logits = build_network()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # If necessary, in the model_fn, use params['batch_size'] instead the batch
    # size flags (--train_batch_size or --eval_batch_size).
    batch_size = params['batch_size']  # pylint: disable=unused-variable

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    one_hot_labels = tf.one_hot(labels, hparams.num_label_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
        label_smoothing=hparams.label_smoothing)

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + hparams.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    host_call = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from global_step.
        global_step = tf.train.get_global_step()
        steps_per_epoch = hparams.num_train_images / hparams.train_batch_size
        current_epoch = (tf.cast(global_step, tf.float32) /
                         steps_per_epoch)
        # LARS is a large batch optimizer. LARS enables higher accuracy at batch 16K
        # and larger batch sizes.

        learning_rate = learning_rate_schedule(hparams.train_steps, current_epoch)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=hparams.momentum,
            use_nesterov=True)
        if hparams.use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

        if not hparams.skip_host_call:
            def host_call_fn(gs, loss, lr, ce, img_grid):
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
                tf.print([tf.reduce_max(img_grid), tf.reduce_min(img_grid), tf.reduce_mean(img_grid)])
                # once per loop.a
                with summary.create_file_writer(
                        hparams.model_dir, max_queue=hparams.iterations_per_loop).as_default():
                    with summary.always_record_summaries():
                        summary.scalar('loss', loss[0], step=gs)
                        summary.scalar('learning_rate', lr[0], step=gs)
                        summary.scalar('current_epoch', ce[0], step=gs)
                        summary.image('img_grid', img_grid, max_images=1,step=gs)

                        return summary.all_summary_ops()

            # To log the loss, current learning rate, and epoch for Tensorboard, the
            # summary op needs to be run on the host CPU via host_call. host_call
            # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
            # dimension. These Tensors are implicitly concatenated to
            # [params['batch_size']].
            gs_t = tf.reshape(global_step, [1])
            loss_t = tf.reshape(loss, [1])
            lr_t = tf.reshape(learning_rate, [1])
            ce_t = tf.reshape(current_epoch, [1])
            img_grid = tf.contrib.gan.eval.image_grid(
                features,
                [16, 8],
                image_shape=(hparams.image_size, hparams.image_size),
                num_channels=3
            )

            host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t, features])

    else:
        train_op = None

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(labels, logits, feature):
            """Evaluation metric function. Evaluates accuracy.

            This function is executed on the CPU and should not directly reference
            any Tensors in the rest of the `model_fn`. To pass Tensors from the model
            to the `metric_fn`, provide as part of the `eval_metrics`. See
            https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
            for more information.

            Arguments should match the list of `Tensor` objects passed as the second
            element in the tuple passed to `eval_metrics`.

            Args:
              labels: `Tensor` with shape `[batch]`.
              logits: `Tensor` with shape `[batch, num_classes]`.

            Returns:
              A dict of the metrics to return from evaluation.
            """
            predictions = tf.argmax(logits, axis=1)
            top_1_accuracy = tf.metrics.accuracy(labels, predictions)
            in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
            top_5_accuracy = tf.metrics.mean(in_top_5)
            # top_5_accuracy = tf.Print(top_5_accuracy, [labels, featur])

            return {
                'top_1_accuracy': top_1_accuracy,
                'top_5_accuracy': top_5_accuracy,
                # 'lmax': tf.reduce_max(labels),
                # 'lmin': tf.reduce_min(labels),
                'fmax': tf.reduce_max(feature),
                'fmin': tf.reduce_min(feature),

            }

        eval_metrics = (metric_fn, [labels, logits, features])


    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        host_call=host_call,
        eval_metrics=eval_metrics)


def _verify_non_empty_string(value, field_name):
    """Ensures that a given proposed field value is a non-empty string.

    Args:
      value:  proposed value for the field.
      field_name:  string name of the field, e.g. `project`.

    Returns:
      The given value, provided that it passed the checks.

    Raises:
      ValueError:  the value is not a string, or is a blank string.
    """
    if not isinstance(value, str):
        raise ValueError(
            'Bigtable parameter "%s" must be a string.' % field_name)
    if not value:
        raise ValueError(
            'Bigtable parameter "%s" must be non-empty.' % field_name)
    return value


def main(unused_argv):

    tpu = 'chocoarthur'
    tpu_zone = 'us-central1-f'
    gcp_project = 'cloud-tpu-epfl'

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu if (hparams.tpu or hparams.use_tpu) else '',
        zone=tpu_zone,
        project=gcp_project)

    if hparams.use_async_checkpointing:
        save_checkpoints_steps = None
    else:
        save_checkpoints_steps = max(100, hparams.iterations_per_loop)

    config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=hparams.model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=hparams.log_step_count_steps,
        session_config=tf.ConfigProto(
            graph_options=tf.GraphOptions(
                rewrite_options=rewriter_config_pb2.RewriterConfig(
                    disable_meta_optimizer=True))),
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=hparams.iterations_per_loop,
            num_shards=hparams.num_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
                .PER_HOST_V2))  # pylint: disable=line-too-long

    resnet_classifier = tf.contrib.tpu.TPUEstimator(
        use_tpu=hparams.use_tpu,
        model_fn=resnet_model_fn,
        config=config,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.eval_batch_size,
        export_to_tpu=hparams.export_to_tpu)
    assert hparams.precision == 'bfloat16' or hparams.precision == 'float32', (
        'Invalid value for --precision flag; must be bfloat16 or float32.')
    tf.logging.info('Precision: %s', hparams.precision)
    use_bfloat16 = hparams.precision == 'bfloat16'

    # Input pipelines are slightly different (with regards to shuffling and
    # preprocessing) between training and evaluation.
    if hparams.data_dir == FAKE_DATA_DIR:
        tf.logging.info('Using fake dataset.')
    else:
        tf.logging.info('Using dataset: %s', hparams.data_dir)
    # imagenet_train, imagenet_eval = [
    #     imagenet_input.ImagenetRecordInput(
    #         is_training=is_training,
    #         data_dir=hparams.data_dir,
    #         transpose_input=hparams.transpose_input,
    #         cache=hparams.use_cache and is_training,
    #         image_size=hparams.image_size,
    #         num_parallel_calls=hparams.num_parallel_calls,
    #         use_bfloat16=use_bfloat16) for is_training in [True, False]
    # ]

    imagenet_train = imagenet_input.InputFunction(
        is_training=True,
        noise_dim=128,
        num_classes=hparams.num_label_classes,
        data_dir=hparams.data_dir,
    )

    imagenet_eval = imagenet_input.InputFunction(
        is_training=False,
        noise_dim=128,
        num_classes=hparams.num_label_classes,
        data_dir=hparams.data_dir,
    )

    eval_steps = hparams.num_eval_images // hparams.eval_batch_size

    if hparams.mode == 'eval':
        # Run evaluation when there's a new checkpoint
        for ckpt in evaluation.checkpoints_iterator(
                model_dir, timeout=hparams.eval_timeout):
            tf.logging.info('Starting to evaluate.')
            try:
                start_timestamp = time.time()  # This time will include compilation time
                eval_results = resnet_classifier.evaluate(
                    input_fn=imagenet_eval.input_fn,
                    steps=eval_steps,
                    checkpoint_path=ckpt)
                elapsed_time = int(time.time() - start_timestamp)
                tf.logging.info('Eval results: %s. Elapsed seconds: %d',
                                eval_results, elapsed_time)

                # Terminate eval job when final checkpoint is reached
                current_step = int(os.path.basename(ckpt).split('-')[1])
                if current_step >= hparams.train_steps:
                    tf.logging.info(
                        'Evaluation finished after training step %d', current_step)
                    break

            except tf.errors.NotFoundError:
                # Since the coordinator is on a different job than the TPU worker,
                # sometimes the TPU worker does not finish initializing until long after
                # the CPU job tells it to start evaluating. In this case, the checkpoint
                # file could have been deleted already.
                tf.logging.info(
                    'Checkpoint %s no longer exists, skipping checkpoint', ckpt)

    else:  # hparams.mode == 'train' or hparams.mode == 'train_and_eval'
        current_step = estimator._load_global_step_from_checkpoint_dir(
            hparams.model_dir)  # pylint: disable=protected-access,line-too-long
        steps_per_epoch = hparams.num_train_images // hparams.train_batch_size

        tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                        ' step %d.',
                        hparams.train_steps,
                        hparams.train_steps / steps_per_epoch,
                        current_step)

        start_timestamp = time.time()  # This time will include compilation time

        if hparams.mode == 'train':
            hooks = []
            if hparams.use_async_checkpointing:
                hooks.append(
                    async_checkpoint.AsyncCheckpointSaverHook(
                        checkpoint_dir=model_dir,
                        save_steps=max(100, hparams.iterations_per_loop)))
            if hparams.profile_every_n_steps > 0:
                hooks.append(
                    tpu_profiler_hook.TPUProfilerHook(
                        save_steps=hparams.profile_every_n_steps,
                        output_dir=model_dir, tpu=hparams.tpu)
                )
            resnet_classifier.train(
                input_fn=imagenet_train,
                max_steps=hparams.train_steps,
                hooks=hooks)

        else:
            assert hparams.mode == 'train_and_eval'
            while current_step < hparams.train_steps:
                # Train for up to steps_per_eval number of steps.
                # At the end of training, a checkpoint will be written to --model_dir.
                next_checkpoint = min(current_step + hparams.steps_per_eval,
                                      hparams.train_steps)
                resnet_classifier.train(
                    input_fn=imagenet_train, max_steps=next_checkpoint)
                current_step = next_checkpoint

                tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                                next_checkpoint, int(time.time() - start_timestamp))

                # Evaluate the model on the most recent model in --model_dir.
                # Since evaluation happens in batches of --eval_batch_size, some images
                # may be excluded modulo the batch size. As long as the batch size is
                # consistent, the evaluated images are also consistent.
                tf.logging.info('Starting to evaluate.')
                eval_results = resnet_classifier.evaluate(
                    input_fn=imagenet_eval,
                    steps=hparams.num_eval_images // hparams.eval_batch_size)
                tf.logging.info('Eval results at step %d: %s',
                                next_checkpoint, eval_results)

            elapsed_time = int(time.time() - start_timestamp)
            tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                            hparams.train_steps, elapsed_time)

        if hparams.export_dir is not None:
            # The guide to serve a exported TensorFlow model is at:
            #    https://www.tensorflow.org/serving/serving_basic
            tf.logging.info('Starting to export model.')
            resnet_classifier.export_saved_model(
                export_dir_base=hparams.export_dir,
                serving_input_receiver_fn=imagenet_input.image_serving_input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
