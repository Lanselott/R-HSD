# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Vgg16 V1 Faster R-CNN implementation.

See "Deep Residual Learning for Image Recognition" by He et al., 2015.
https://arxiv.org/abs/1512.03385

Note: this implementation assumes that the classification checkpoint used
to finetune this model is trained using the same configuration as that of
the MSRA provided checkpoints
(see https://github.com/KaimingHe/deep-residual-networks), e.g., with
same preprocessing, batch norm scaling, etc.
"""
import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import vgg
#from nets import Vgg16_utils
#from nets import Vgg16_v1

slim = tf.contrib.slim

class FasterRCNNVggFeatureExtractor(
  faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Vgg 16 feature extractor implementation."""

  def __init__(self,
                architecture,
                vgg_model,
                is_training,
                first_stage_features_stride,
                batch_norm_trainable=False,
                reuse_weights=None,
                weight_decay=0.0):
      """Constructor.

      Args:
        architecture: Architecture name of the Vgg model.
        vgg16_model: Definition of the Vgg  model.
        is_training: See base class.
        first_stage_features_stride: See base class.
        batch_norm_trainable: See base class.
        reuse_weights: See base class.
        weight_decay: See base class.

      Raises:
        ValueError: If `first_stage_features_stride` is not 8 or 16.
      """
      if first_stage_features_stride != 8 and first_stage_features_stride != 16:
        raise ValueError('`first_stage_features_stride` must be 8 or 16.')
      self._architecture = architecture
      self._vgg_model = vgg_model
      super(FasterRCNNVggFeatureExtractor, self).__init__(
          is_training, first_stage_features_stride, batch_norm_trainable,
          reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Vgg16 preprocessing.

    VGG style channel mean subtraction as described here:
      https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

      Args:
        resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
          representing a batch of images with values between 0 and 255.0.

      Returns:
        preprocessed_inputs: A [batch, height_out, width_out, channels] float32
          tensor representing a batch of images.

    """ 
    channel_means = [123.68, 116.779, 103.939]
    return resized_inputs - [[channel_means]]

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

      Args:
        preprocessed_inputs: A [batch, height, width, channels] float32 tensor
          representing a batch of images.
        scope: A scope name.

      Returns:
        rpn_feature_map: A tensor with shape [batch, height, width, depth]
        activations: A dictionary mapping feature extractor tensor names to
          tensors
 
      Raises:
        InvalidArgumentError: If the spatial size of `preprocessed_inputs`
          (height or width) is less than 33.
        ValueError: If the created network is missing the required activation.
      """
    
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    # Need to update shape_assert for correct size in VGG16
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
            tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
        with tf.variable_scope(
          self._architecture, reuse=self._reuse_weights) as var_scope:
          _, activations = self._vgg_model(
              preprocessed_inputs,
              num_classes=None,
              is_training=self._train_batch_norm,
              dropout_keep_prob=1,
              spatial_squeeze=False,
              scope=var_scope,
              fc_conv_padding='VALID',
              global_pool=False,
              end_point = 'pool5')
    ''' CHEN: handle should be update if not vgg16'''
    handle = scope + '/%s/vgg_16/pool5' % self._architecture
    print("activations:",activations)
    return activations[handle], activations

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    net = proposal_feature_maps
    
    with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
      with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, 1, is_training=True,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, 1, is_training=True,
                           scope='dropout7')
        net = slim.conv2d(net, 3, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        proposal_classifier_features = net

    return proposal_classifier_features

class FasterRCNNVgg16FeatureExtractor(FasterRCNNVggFeatureExtractor):
  """VGG 16 feature extractor"""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16,
        or if `architecture` is not supported.
    """
    super(FasterRCNNVgg16FeatureExtractor, self).__init__(
        'vgg_16', vgg.vgg_16, is_training,
        first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)