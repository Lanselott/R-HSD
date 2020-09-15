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
"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import collections
import functools
# Set headless-friendly backend.
import matplotlib
matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf
import time
import pandas as pd

import math
from numpy import genfromtxt
from object_detection.core import standard_fields as fields
# CHEN: shapely: for false alarm/accuracy area union and intersection operation
from shapely.ops import cascaded_union
import shapely.geometry.geo as geo

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige',
    'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk',
    'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
    'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise',
    'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick',
    'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold',
    'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory',
    'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon',
    'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow',
    'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon',
    'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey',
    'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
    'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen',
    'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise',
    'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite',
    'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid',
    'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue',
    'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat',
    'White', 'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def save_image_array_as_png(image, output_path):
    """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.gfile.Open(output_path, 'w') as fid:
        image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
    """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
    image_pil = Image.fromarray(np.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                 display_str_list_list)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
    """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1],
                                   boxes[i, 2], boxes[i, 3], color, thickness,
                                   display_str_list)


def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_boxes_and_masks(image, boxes, classes, scores, masks,
                               category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        **kwargs)


def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints,
                                   category_index, **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        keypoints=keypoints,
        **kwargs)


def _visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores,
                                             masks, keypoints, category_index,
                                             **kwargs):
    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        keypoints=keypoints,
        **kwargs)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         instance_masks=None,
                                         keypoints=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
    """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored.
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
    # Additional channels are being ignored.
    images = images[:, :, :, 0:3]
    visualization_keyword_args = {
        'use_normalized_coordinates': use_normalized_coordinates,
        'max_boxes_to_draw': max_boxes_to_draw,
        'min_score_thresh': min_score_thresh,
        'agnostic_mode': False,
        'line_thickness': 4
    }

    if instance_masks is not None and keypoints is None:
        visualize_boxes_fn = functools.partial(_visualize_boxes_and_masks,
                                               category_index=category_index,
                                               **visualization_keyword_args)
        elems = [images, boxes, classes, scores, instance_masks]
    elif instance_masks is None and keypoints is not None:
        visualize_boxes_fn = functools.partial(_visualize_boxes_and_keypoints,
                                               category_index=category_index,
                                               **visualization_keyword_args)
        elems = [images, boxes, classes, scores, keypoints]
    elif instance_masks is not None and keypoints is not None:
        visualize_boxes_fn = functools.partial(
            _visualize_boxes_and_masks_and_keypoints,
            category_index=category_index,
            **visualization_keyword_args)
        elems = [images, boxes, classes, scores, instance_masks, keypoints]
    else:
        visualize_boxes_fn = functools.partial(_visualize_boxes,
                                               category_index=category_index,
                                               **visualization_keyword_args)
        elems = [images, boxes, classes, scores]

    def draw_boxes(image_and_detections):
        """Draws boxes on image."""
        image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections,
                                      tf.uint8)
        return image_with_boxes

    images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
    return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True):
    """Creates a side-by-side image with detections and groundtruth.

  Bounding boxes (and instance masks, if available) are visualized on both
  subimages.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_single_example().
    category_index: A category index (dictionary) produced from a labelmap.
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    A [1, H, 2 * W, C] uint8 tensor. The subimage on the left corresponds to
      detections, while the subimage on the right corresponds to groundtruth.
  """
    detection_fields = fields.DetectionResultFields()
    input_data_fields = fields.InputDataFields()
    instance_masks = None
    if detection_fields.detection_masks in eval_dict:
        instance_masks = tf.cast(
            tf.expand_dims(eval_dict[detection_fields.detection_masks],
                           axis=0), tf.uint8)
    keypoints = None
    if detection_fields.detection_keypoints in eval_dict:
        keypoints = tf.expand_dims(
            eval_dict[detection_fields.detection_keypoints], axis=0)
    groundtruth_instance_masks = None
    if input_data_fields.groundtruth_instance_masks in eval_dict:
        groundtruth_instance_masks = tf.cast(
            tf.expand_dims(
                eval_dict[input_data_fields.groundtruth_instance_masks],
                axis=0), tf.uint8)
    images_with_detections = draw_bounding_boxes_on_image_tensors(
        eval_dict[input_data_fields.original_image],
        tf.expand_dims(eval_dict[detection_fields.detection_boxes], axis=0),
        tf.expand_dims(eval_dict[detection_fields.detection_classes], axis=0),
        tf.expand_dims(eval_dict[detection_fields.detection_scores], axis=0),
        category_index,
        instance_masks=instance_masks,
        keypoints=keypoints,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates)
    images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
        eval_dict[input_data_fields.original_image],
        tf.expand_dims(eval_dict[input_data_fields.groundtruth_boxes], axis=0),
        tf.expand_dims(eval_dict[input_data_fields.groundtruth_classes],
                       axis=0),
        tf.expand_dims(
            tf.ones_like(eval_dict[input_data_fields.groundtruth_classes],
                         dtype=tf.float32),
            axis=0),
        category_index,
        instance_masks=groundtruth_instance_masks,
        keypoints=None,
        max_boxes_to_draw=None,
        min_score_thresh=0.0,
        use_normalized_coordinates=use_normalized_coordinates)
    return tf.concat([images_with_detections, images_with_groundtruth], axis=2)


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
    """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius,
                            use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
    """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                      (keypoint_x + radius, keypoint_y + radius)],
                     outline=color,
                     fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError(
            'The image has spatial dimensions %s but the mask has '
            'dimensions %s' % (image.shape[:2], mask.shape))
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(
        list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


hotspot_list = []
class_1_list = []
class_2_list = []
class_3_list = []
groundtruth_list = []
class1_groundtruth_list = []
class2_groundtruth_list = []
class3_groundtruth_list = []

eval_counter = 0
num_visualize = 0

false_alarm_list = []
false_alarm_1_list = []
false_alarm_2_list = []
false_alarm_3_list = []
start_timer = 0
timer_1 = 0
timer_2 = 0
tiemr_3 = 0

pred_gt_distance = []
predict_bbox_set = []
predict_cls_list = []
predict_bbox_list = []


def eval_hotspot(classes,
                 scores,
                 category_index,
                 groundtruth_boxes,
                 detection_boxes,
                 min_score_thresh=.5,
                 num_examples=100):
    global eval_counter
    global timer_1
    global timer_2
    global timer_3
    global start_timer
    global num_visualize
    """
    CHEN: here we calculate the centre point of groundtruth & detection hotspot
    if the distance between the point larger than the 1/3 of the crop iamge size
    than it will considered as mismatch
    """
    offset = 23

    groundtruth_count_flag = np.ones(groundtruth_boxes.shape[0])
    hotspot = 0

    for i in range(detection_boxes.shape[0]):
        if eval_counter == 0:
            start_timer = time.time()
        elif eval_counter == 7:
            timer_1 = time.time()
        elif eval_counter == 40:
            timer_2 = time.time()
        elif eval_counter == 95:
            timer_3 = time.time()
            # print("case 1 GPU time:",timer_1 - start_timer)
            # print("case 2 GPU time:",timer_2 - timer_1)
            # print("case 3 GPU time:",timer_3 - timer_2)
        '''
      R-HSD settings: 0.4/ 0.6/ 0.6
      '''

        if eval_counter < 8:
            min_score_thresh = 0.2  #0.8#0.80
            #min_score_thresh = 0.62
        if eval_counter < 41 and eval_counter >= 8:
            min_score_thresh = 0.60
            #min_score_thresh = 0.4
        if eval_counter < 96 and eval_counter >= 41:
            min_score_thresh = 0.60  #0.62 #in case three modify the threshold to ensure the balance between acc and hotspot
            #min_score_thresh = 0.4

        if scores[i] > min_score_thresh:
            flag = 0

            detected_box = tuple(detection_boxes[i].tolist())
            #print("eval_hotspot:box:",box[0],box[1],box[2],box[3])
            detection_hotspot_x = (detected_box[2] -
                                   detected_box[0]) / 2 + detected_box[0]
            detection_hotspot_y = (detected_box[3] -
                                   detected_box[1]) / 2 + detected_box[1]

            #print("detected hotspot point:",detection_hotspot_x,detection_hotspot_y)
            for j in range(groundtruth_boxes.shape[0]):
                ground_box = tuple(groundtruth_boxes[j].tolist())
                groundtruth_hotspot_x = (ground_box[2] -
                                         ground_box[0]) / 2 + ground_box[0]
                groundtruth_hotspot_y = (ground_box[3] -
                                         ground_box[1]) / 2 + ground_box[1]
                dis_x_threshold = (ground_box[2] - ground_box[0]) / 6
                dis_y_threshold = (ground_box[3] - ground_box[1]) / 6
                '''special case 4:'''
                if classes[i] == 3:
                    dis_x_threshold = (ground_box[2] - ground_box[0]) / 3
                    dis_y_threshold = (ground_box[3] - ground_box[1]) / 3
                #print("dis_x_threshold:",dis_x_threshold)
                x_axis_distance = abs(groundtruth_hotspot_x -
                                      detection_hotspot_x)
                y_axis_distance = abs(groundtruth_hotspot_y -
                                      detection_hotspot_y)
                predict_bbox_set.append(
                    [detection_hotspot_x / 256.0, detection_hotspot_y / 256.0])
                if x_axis_distance <= dis_x_threshold and \
                  y_axis_distance <= dis_y_threshold:
                    pred_gt_distance.append(
                        math.sqrt(x_axis_distance * x_axis_distance +
                                  y_axis_distance * y_axis_distance))

                    hotspot += 1
                    flag = 1
                    groundtruth_count_flag[j] = 0
                    # print("detected count:",hotspot)
                    continue

            if flag == 0:
                if eval_counter < 8:
                    false_alarm_1_list.append(1)
                elif eval_counter >= 8 and eval_counter < 41:
                    false_alarm_2_list.append(1)
                elif eval_counter >= 41 and eval_counter < 96:
                    false_alarm_3_list.append(1)
    # inference
    if sum([score > min_score_thresh for score in scores]) == 0:
        predict_cls_list.append(0)
        predict_bbox_list.append(['None'])
    else:
        predict_cls_list.append(1)
        predict_bbox_list.append(predict_bbox_set)

    miss_detected = np.count_nonzero(groundtruth_count_flag)

    if eval_counter < 8:
        class1_miss_detected = miss_detected
        class_1_list.append(groundtruth_boxes.shape[0] - class1_miss_detected)
        class1_groundtruth_list.append(groundtruth_boxes.shape[0])
    elif eval_counter >= 8 and eval_counter < 41:
        class2_miss_detected = miss_detected
        class_2_list.append(groundtruth_boxes.shape[0] - class2_miss_detected)
        class2_groundtruth_list.append(groundtruth_boxes.shape[0])
    elif eval_counter >= 41 and eval_counter < 96:
        class3_miss_detected = miss_detected
        class_3_list.append(groundtruth_boxes.shape[0] - class3_miss_detected)
        class3_groundtruth_list.append(groundtruth_boxes.shape[0])

    #accuracy and false alarm for each crop image
    hotspot_list.append(groundtruth_boxes.shape[0] - miss_detected)

    groundtruth_list.append(groundtruth_boxes.shape[0])

    class1_acc = 0
    class2_acc = 0
    class3_acc = 0

    gt1 = float(sum(class1_groundtruth_list))
    gt2 = float(sum(class2_groundtruth_list))
    gt3 = float(sum(class3_groundtruth_list))

    tp = float(sum(hotspot_list))
    tp1 = float(sum(class_1_list))
    tp2 = float(sum(class_2_list))
    tp3 = float(sum(class_3_list))

    fp1 = float(sum(false_alarm_1_list))
    fp2 = float(sum(false_alarm_2_list))
    fp3 = float(sum(false_alarm_3_list))
    fp = fp1 + fp2 + fp3

    fn1 = gt1 - tp1
    fn2 = gt2 - tp2
    fn3 = gt3 - tp3
    fn = fn1 + fn2 + fn3

    accuracy = float(tp) / float(sum(groundtruth_list) + 1)
    if float(gt1) != 0:
        class1_acc = float(tp1) / float(gt1)
    if float(gt3) != 0:
        class2_acc = float(tp2) / float(gt2)
    if float(gt3) != 0:
        class3_acc = float(tp3) / float(gt3)

    f1_score1 = 1
    f1_score2 = 1
    f1_score3 = 1
    f1_score = 1

    fn1 = gt1 - tp1
    fn2 = gt2 - tp2
    fn3 = gt3 - tp3
    ''''''
    # avg_distance = sum(pred_gt_distance) / len(pred_gt_distance)
    if tp1 != 0 and fp1 != 0:
        prec1 = tp1 / (tp1 + fp1)
        recall1 = tp1 / (tp1 + fn1)
        f1_score1 = 2 * (prec1 * recall1) / (prec1 + recall1)
    if tp2 != 0 and fp2 != 0:
        prec2 = tp2 / (tp2 + fp2)
        recall2 = tp2 / (tp2 + fn2)
        f1_score2 = 2 * (prec2 * recall2) / (prec2 + recall2)
    if tp3 != 0 and fp3 != 0:
        prec3 = tp3 / (tp3 + fp3)
        recall3 = tp3 / (tp3 + fn3)
        f1_score3 = 2 * (prec3 * recall3) / (prec3 + recall3)
    if tp != 0 and fp != 0:
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (prec * recall) / (prec + recall)

    sum_false_alarm = sum(false_alarm_list)

    if (num_visualize + 1) == num_examples:
        print("predict cls list:", predict_cls_list)
        print("predict boxes list:", predict_bbox_list)
        cls_result = pd.DataFrame(predict_cls_list)
        bbox_result = pd.DataFrame(predict_bbox_list)

        cls_result.to_csv("cls_result.csv")
        bbox_result.to_csv("bbox_result.csv")
    #   print("count of GT of hotspot:{} \r".format(sum(groundtruth_list)))
    #   print("count of DETECTED hotspot:{} \r".format(tp))
    #   print("class 1 GT count:{} \r".format(sum(class1_groundtruth_list)))
    #   print("class 2 GT count:{} \r".format(sum(class2_groundtruth_list)))
    #   print("class 3 GT count:{} \r".format(sum(class3_groundtruth_list)))
    #   print("ACCURACY:{} \r".format(accuracy))
    #   print("class1_acc:{} \r".format(class1_acc))
    #   print("class2_acc:{} \r".format(class2_acc))
    #   print("class3_acc:{} \r".format(class3_acc))

    #   print("class 1 False Alarm:{} \r".format(fp1))
    #   print("class 2 False Alarm:{} \r".format(fp2))
    #   print("class 3 False Alarm:{} \r".format(fp3))

    #   print("f1 scores for class 1:{} \r".format(f1_score1))
    #   print("f1 scores for class 2:{} \r".format(f1_score2))
    #   print("f1 scores for class 3:{} \r".format(f1_score3))
    #   print("f1 scores for ALL:{} \r".format(f1_score))

    #   print("TIMER :{} \r".format(time.time() - start_timer))

    if len(category_index) == 3:  # ICCAD 16
        eval_counter += 1
    else:  # via data ICCAD12
        eval_counter = 1

    num_visualize += 1
    # return hotspot,sum_false_alarm,miss_detected


def one_stage_eval_hotspot(classes,
                           scores,
                           category_index,
                           groundtruth_boxes,
                           detection_boxes,
                           min_score_thresh=.5):
    """
    CHEN: here we calculate the centre point of groundtruth & detection hotspot
    if the distance between the point larger than the 1/3 of the crop iamge size
    than it will considered as mismatch
    CHEN: For one stage accuracy
    """
    global eval_counter
    global timer_1

    global start_timer

    groundtruth_count_flag = np.ones(groundtruth_boxes.shape[0])
    hotspot = 0
    false_alarm = 0

    for i in range(detection_boxes.shape[0]):
        '''
      if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
              print("class:",class_name)
      '''
        #print("detection_boxes.shape[0]:",detection_boxes.shape[0])
        if eval_counter == 0:
            start_timer = time.time()
        elif eval_counter == 95:
            timer_1 = time.time()

            print("case all GPU time:", timer_1 - start_timer)

        min_score_thresh = 0.99
        if classes[i] != 1 and classes[i] != 0 and classes[i] != 2 and classes[
                i] != 3:
            print("ERROR classfication")

        if scores[i] > min_score_thresh and classes[i] == 1:
            flag = 0

            detected_box = tuple(detection_boxes[i].tolist())
            #print("eval_hotspot:box:",box[0],box[1],box[2],box[3])
            detection_hotspot_x = (detected_box[2] -
                                   detected_box[0]) / 2 + detected_box[0]
            detection_hotspot_y = (detected_box[3] -
                                   detected_box[1]) / 2 + detected_box[1]
            #print("detected hotspot point:",detection_hotspot_x,detection_hotspot_y)
            for j in range(groundtruth_boxes.shape[0]):
                ground_box = tuple(groundtruth_boxes[j].tolist())
                groundtruth_hotspot_x = (ground_box[2] -
                                         ground_box[0]) / 2 + ground_box[0]
                groundtruth_hotspot_y = (ground_box[3] -
                                         ground_box[1]) / 2 + ground_box[1]
                dis_x_threshold = (ground_box[2] - ground_box[0]) / 6
                dis_y_threshold = (ground_box[3] - ground_box[1]) / 6
                '''special case 4:'''
                if classes[i] == 3:
                    dis_x_threshold = (ground_box[2] - ground_box[0]) / 3
                    dis_y_threshold = (ground_box[3] - ground_box[1]) / 3
                #print("dis_x_threshold:",dis_x_threshold)
                if abs(groundtruth_hotspot_x -
                       detection_hotspot_x) <= dis_x_threshold and abs(
                           groundtruth_hotspot_y -
                           detection_hotspot_y) <= dis_y_threshold:
                    hotspot += 1
                    flag = 1
                    groundtruth_count_flag[j] = 0
                    #print("detected count:",hotspot)
                    continue
            if flag == 0:
                false_alarm += 1

    #miss_detected = groundtruth_boxes.shape[0] - hotspot
    miss_detected = np.count_nonzero(groundtruth_count_flag)
    #class1_miss_detected = 0
    #class2_miss_detected = 0
    #class3_miss_detected = 0

    class1_miss_detected = miss_detected
    class_1_list.append(groundtruth_boxes.shape[0] - class1_miss_detected)
    class1_groundtruth_list.append(groundtruth_boxes.shape[0])

    #accuracy and false alarm for each crop image
    #accuracy = (float(groundtruth_boxes.shape[0]) - float(miss_detected)) / float(groundtruth_boxes.shape[0])
    hotspot_list.append(groundtruth_boxes.shape[0] - miss_detected)

    false_alarm_list.append(false_alarm)
    groundtruth_list.append(groundtruth_boxes.shape[0])

    class1_acc = 0

    accuracy = float(sum(hotspot_list)) / float(sum(groundtruth_list))
    if float(sum(class1_groundtruth_list)) != 0:
        class1_acc = float(sum(class_1_list)) / float(
            sum(class1_groundtruth_list))

    sum_false_alarm = sum(false_alarm_list)
    print("count of GT of hotspot:", sum(groundtruth_list))
    print("count of DETECTED hotspot:", sum(hotspot_list))
    print("class 1 GT count:", sum(class1_groundtruth_list))
    print("ACCURACY:", accuracy)
    print("class1_acc:", class1_acc)
    print("FALSE ALARM:", sum_false_alarm)
    eval_counter += 1

    return hotspot, false_alarm, miss_detected


#sum_acc_area = []
#sum_false_alarm_area = []
sum_gt_area = []
sum_detection_area = []
sum_FA_detection_area = []


def eval_area_hotspot(classes,
                      scores,
                      category_index,
                      groundtruth_boxes,
                      detection_boxes,
                      min_score_thresh=.5):
    """
    CHEN: calculate the accuracy and false alarm base on area
    gt_area: ground truth area(box) of the hotspot
    sum_detection_area: hotspot detected area(NOT false alarm)
    sum_FA_detection_area: hotspot detected area(false alarm)
    """

    groundtruth_count_flag = np.ones(groundtruth_boxes.shape[0])
    hotspot = 0
    false_alarm = 0

    count_each_label = False

    acc_area = 0
    false_alarm_area = 0
    gt_area = 0

    acc_lx = 0
    acc_ly = 0
    acc_hx = 0
    acc_hy = 0

    for i in range(detection_boxes.shape[0]):
        '''
      if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
              print("class:",class_name)
      '''
        #print("detection_boxes.shape[0]:",detection_boxes.shape[0])
        if classes[i] == 1:
            min_score_thresh = 0.75
        if classes[i] == 2:
            min_score_thresh = 0.95
        if classes[i] == 3:
            min_score_thresh = 0.70

        if scores[i] > min_score_thresh:
            flag = 0

            detected_box = tuple(detection_boxes[i].tolist())
            detected_box_0 = int(detected_box[0])
            detected_box_1 = int(detected_box[1])
            detected_box_2 = int(detected_box[2])
            detected_box_3 = int(detected_box[3])

            #print("type of detected_box[2]:",type(detected_box[2]))
            #print("eval_hotspot:box:",box[0],box[1],box[2],box[3])
            detection_hotspot_x = (detected_box_2 -
                                   detected_box_0) / 2 + detected_box_0
            detection_hotspot_y = (detected_box_3 -
                                   detected_box_1) / 2 + detected_box_1
            #print("detected hotspot point:",detection_hotspot_x,detection_hotspot_y)
            for j in range(groundtruth_boxes.shape[0]):
                ground_box = tuple(groundtruth_boxes[j].tolist())
                ground_box_0 = int(ground_box[0])
                ground_box_1 = int(ground_box[1])
                ground_box_2 = int(ground_box[2])
                ground_box_3 = int(ground_box[3])

                groundtruth_hotspot_x = (ground_box_2 -
                                         ground_box_0) / 2 + ground_box_0
                groundtruth_hotspot_y = (ground_box_3 -
                                         ground_box_1) / 2 + ground_box_1
                dis_x_threshold = (ground_box_2 - ground_box_0) / 3
                dis_y_threshold = (ground_box_3 - ground_box_1) / 3
                if abs(groundtruth_hotspot_x -
                       detection_hotspot_x) <= dis_x_threshold and abs(
                           groundtruth_hotspot_y -
                           detection_hotspot_y) <= dis_y_threshold:
                    '''
            find the pair of [ground_truth box, predicted box]
            '''
                    '''
            if ground_box[2] < detected_box[2]:
              acc_hx = ground_box[2]
            else:
              acc_hx = detected_box[2]

            if ground_box[3] < detected_box[3]:
              acc_hy = ground_box[3]
            else:
              acc_hy = detected_box[3]

            if ground_box[0] < detected_box[0]:
              acc_lx = detected_box[0]
            else:
              acc_lx = ground_box[0]

            if ground_box[1] < detected_box[1]:
              acc_ly = detected_box[1]
            else:
              acc_ly = ground_box[1]

            acc_area = (acc_hx - acc_lx) * (acc_hy - acc_ly)
            sum_acc_area = sum_acc_area + acc_area

            false_alarm_area = (detected_box[3] - detected_box[1]) * (detected_box[2] - detected_box[0]) - acc_area
            sum_false_alarm_area = sum_false_alarm_area + false_alarm_area
            '''
                    gt_area = (ground_box_3 - ground_box_1) * (ground_box_2 -
                                                               ground_box_0)
                    detected_area = (detected_box_3 - detected_box_1) * (
                        detected_box_2 - detected_box_0)
                    sum_gt_area.append(gt_area)

                    sum_detection_area.append(detected_area)
                    sum_FA_detection_area.append()

                    hotspot += 1
                    flag = 1
                    groundtruth_count_flag[j] = 0

                    #print("detected count:",hotspot)
                    continue
            if flag == 0:
                false_alarm += 1
                sum_FA_detection_area.append(
                    (detected_box_3 - detected_box_1) *
                    (detected_box_2 - detected_box_0))

    #print("Sum of detection area:",sum(sum_detection_area))
    #print("Sum of false alarm area:",sum(sum_FA_detection_area))

    miss_detected = np.count_nonzero(groundtruth_count_flag)

    #accuracy and false alarm for each crop image
    #accuracy = (float(groundtruth_boxes.shape[0]) - float(miss_detected)) / float(groundtruth_boxes.shape[0])
    hotspot_list.append(groundtruth_boxes.shape[0] - miss_detected)

    false_alarm_list.append(false_alarm)
    groundtruth_list.append(groundtruth_boxes.shape[0])
    accuracy = float(sum(hotspot_list)) / float(sum(groundtruth_list))
    sum_false_alarm = sum(false_alarm_list)

    if count_each_label:
        if classes[0] == 1:
            class1_miss_detected = miss_detected
            class_1_list.append(groundtruth_boxes.shape[0] -
                                class1_miss_detected)
            class1_groundtruth_list.append(groundtruth_boxes.shape[0])
        elif classes[0] == 2:
            class2_miss_detected = miss_detected
            class_2_list.append(groundtruth_boxes.shape[0] -
                                class2_miss_detected)
            class2_groundtruth_list.append(groundtruth_boxes.shape[0])
        elif classes[0] == 3:
            class3_miss_detected = miss_detected
            class_3_list.append(groundtruth_boxes.shape[0] -
                                class3_miss_detected)
            class3_groundtruth_list.append(groundtruth_boxes.shape[0])

        class1_acc = 0
        class2_acc = 0
        class3_acc = 0

        if float(sum(class1_groundtruth_list)) != 0:
            class1_acc = float(sum(class_1_list)) / float(
                sum(class1_groundtruth_list))
        if float(sum(class2_groundtruth_list)) != 0:
            class2_acc = float(sum(class_2_list)) / float(
                sum(class2_groundtruth_list))
        if float(sum(class3_groundtruth_list)) != 0:
            class3_acc = float(sum(class_3_list)) / float(
                sum(class3_groundtruth_list))
        '''
      print("count of GT of hotspot:",sum(groundtruth_list))
      print("count of DETECTED hotspot:",sum(hotspot_list))
      print("class 1 GT count:",sum(class1_groundtruth_list))
      print("class 2 GT count:",sum(class2_groundtruth_list))
      print("class 3 GT count:",sum(class3_groundtruth_list))
      print("class1_acc:",class1_acc)
      print("class2_acc:",class2_acc)
      print("class3_acc:",class3_acc)
      '''

    #print("ACCURACY:",accuracy)
    #print("FALSE ALARM:",sum_false_alarm)

    return accuracy, sum_false_alarm, sum_detection_area, sum_FA_detection_area, sum_gt_area


gt_area_counter = []
detected_area_counter = []
false_alarm_area_counter = []
intersection_area_counter = []

gt_area_counter_1 = []
detected_area_counter_1 = []
false_alarm_area_counter_1 = []
intersection_area_counter_1 = []

gt_area_counter_2 = []
detected_area_counter_2 = []
false_alarm_area_counter_2 = []
intersection_area_counter_2 = []

gt_area_counter_3 = []
detected_area_counter_3 = []
false_alarm_area_counter_3 = []
intersection_area_counter_3 = []
num_counter = 0
box_saver = []


def eval_area_hotspot_v2(classes,
                         scores,
                         category_index,
                         groundtruth_boxes,
                         detection_boxes,
                         min_score_thresh=.5):
    # gt_area --> coordinate info of ground_truth boxes in one image
    # detected_area --> coordinate info of detected_area boxes in one image
    coord_path = '/research/byu2/rchen/proj/gds'

    csv_file = coord_path + '/shift_saver_' + str(classes[0] + 1) + '.csv'
    coord_array = genfromtxt(csv_file, delimiter=',')  #[X,2]

    global num_counter

    gt_area = []
    detected_area = []
    '''
    label info:
    label1[height,width]:		[695,375]
    label2[height,width]:		[1291,1007]
    label3[height,width]:		[7995,4213]

    label1 step = 256
    label2 step = 256
    label3 step = 128

    num_of_label1 = 6 (3*2)
    num_of_label2 = 24 (6*4)
    num_of_label3 = 55 ()

    follow the eval generator's policy to remap,
    * the actual box is the middle 1/3 part of the detected_box
    *
    -------------------> x direction
    |
    |             *(xh,yh)
    |
    |
    |     *(xl,yl)
    |
    v
    y direction
    '''

    #width = 256
    #if classes[0] == 3:
    #  width = 128

    for i in range(detection_boxes.shape[0]):
        #print("detection_boxes.shape[0]:",detection_boxes.shape[0])
        if classes[i] == 1:
            min_score_thresh = 0.90
        if classes[i] == 2:
            min_score_thresh = 0.90
        if classes[i] == 3:
            min_score_thresh = 0.70

        if scores[i] > min_score_thresh:
            flag = 0
            detected_box = tuple(detection_boxes[i].tolist())

            lx = detected_box[1]
            ly = detected_box[0]
            hx = detected_box[3]
            hy = detected_box[2]

            if classes[0] == 3:
                lx = int(lx / 2)
                ly = int(ly / 2)
                hx = int(hx / 2)
                hy = int(hy / 2)
            '''NOTE: the coordinates are different'''
            '''
        temp = ly
        ly = hy
        hy = temp

        ly = width - ly
        hy = width - hy
        '''
            #print("coord_array shape:",np.shape(coord_array))
            detected_box_lx = int(lx) + coord_array[num_counter, 0]
            detected_box_ly = int(ly) + coord_array[num_counter, 1]
            detected_box_hx = int(hx) + coord_array[num_counter, 0]
            detected_box_hy = int(hy) + coord_array[num_counter, 1]

            #append the detected_box coordinate to the list

            #crop middle 1/3 part as true box
            true_detected_box_lx = detected_box_lx + int(
                (detected_box_hx - detected_box_lx) / 3)
            true_detected_box_hx = detected_box_hx - int(
                (detected_box_hx - detected_box_lx) / 3)
            true_detected_box_ly = detected_box_ly + int(
                (detected_box_hy - detected_box_ly) / 3)
            true_detected_box_hy = detected_box_hy - int(
                (detected_box_hy - detected_box_ly) / 3)

            box_saver.append([
                true_detected_box_lx, true_detected_box_ly,
                true_detected_box_hx, true_detected_box_hy
            ])

    num_counter = num_counter + 1
    '''
    for j in range(groundtruth_boxes.shape[0]):
      ground_box = tuple(groundtruth_boxes[j].tolist())
      ground_box_lx = int(ground_box[0])
      ground_box_ly = int(ground_box[1])
      ground_box_hx = int(ground_box[2])
      ground_box_hy = int(ground_box[3])

      #print("ground_box_lx",ground_box_lx)
      #print("ground_box_ly",ground_box_ly)
      #print("ground_box_hx",ground_box_hx)
      #print("ground_box_hy",ground_box_hy)
      #append the gt_box coordinate to the list
      gt_area.append(geo.box(ground_box_lx,ground_box_ly,ground_box_hx,ground_box_hy))

    ground_poly_region = cascaded_union(gt_area)
    detected_poly_region = cascaded_union(detected_area)

    ground_poly_region_area = ground_poly_region.area
    detected_poly_region_area = detected_poly_region.area

    intersection = ground_poly_region.intersection(detected_poly_region)
    intersection_area = intersection.area
    false_alarm_area = (ground_poly_region_area - intersection_area) + (detected_poly_region_area - intersection_area)

    #DIVIDE counter
    if classes[0] == 1:
      gt_area_counter_1.append(ground_poly_region_area)
      detected_area_counter_1.append(detected_poly_region_area)
      false_alarm_area_counter_1.append(false_alarm_area)
      intersection_area_counter_1.append(intersection_area)
      print("LABEL ONE:Sum of ground truth area:",sum(gt_area_counter_1))
      print("LABEL ONE:Sum of detected area:",sum(detected_area_counter_1))
      print("LABEL ONE:Sum of intersection area:",sum(intersection_area_counter_1))
      print("LABEL ONE:Sum of false alarm area:",sum(false_alarm_area_counter_1))
      print("LABEL ONE:Area accuracy:",float(sum(intersection_area_counter_1))/float(sum(gt_area_counter_1)))
    elif classes[0] == 2:
      gt_area_counter_2.append(ground_poly_region_area)
      detected_area_counter_2.append(detected_poly_region_area)
      false_alarm_area_counter_2.append(false_alarm_area)
      intersection_area_counter_2.append(intersection_area)
      print("LABEL TWO:Sum of ground truth area:",sum(gt_area_counter_2))
      print("LABEL TWO:Sum of detected area:",sum(detected_area_counter_2))
      print("LABEL TWO:Sum of intersection area:",sum(intersection_area_counter_2))
      print("LABEL TWO:Sum of false alarm area:",sum(false_alarm_area_counter_2))
      print("LABEL TWO:Area accuracy:",float(sum(intersection_area_counter_2))/float(sum(gt_area_counter_2)))
    elif classes[0] == 3:
      gt_area_counter_3.append(ground_poly_region_area)
      detected_area_counter_3.append(detected_poly_region_area)
      false_alarm_area_counter_3.append(false_alarm_area)
      intersection_area_counter_3.append(intersection_area)
      print("LABEL THREE:Sum of ground truth area:",sum(gt_area_counter_3))
      print("LABEL THREE:Sum of detected area:",sum(detected_area_counter_3))
      print("LABEL THREE:Sum of intersection area:",sum(intersection_area_counter_3))
      print("LABEL THREE:Sum of false alarm area:",sum(false_alarm_area_counter_3))
      print("LABEL THREE:Area accuracy:",float(sum(intersection_area_counter_3))/float(sum(gt_area_counter_3)))


    #ALL counter
    gt_area_counter.append(ground_poly_region_area)
    detected_area_counter.append(detected_poly_region_area)
    false_alarm_area_counter.append(false_alarm_area)
    intersection_area_counter.append(intersection_area)
    '''
    return box_saver, num_counter


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.9,
        agnostic_mode=False,
        line_thickness=3,
        groundtruth_box_visualization_color='red',
        skip_scores=False,
        skip_labels=False):
    """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            #print("610:box:",box)
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100 * scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str,
                                                       int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(image,
                                     box_to_instance_masks_map[box],
                                     color=color)
        if instance_boundaries is not None:
            draw_mask_on_image_array(image,
                                     box_to_instance_boundaries_map[box],
                                     color='green',
                                     alpha=0.5)
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color='red',
            thickness=line_thickness,
            display_str_list=
            [],  #box_to_display_str_map[box], #CHEN: remove the string for better visualize on hotpsot task
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates)

    return image


def add_cdf_image_summary(values, name):
    """Adds a tf.summary.image for a CDF plot of the values.

  Normalizes `values` such that they sum to 1, plots the cumulative distribution
  function and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    name: name for the image summary.
  """
    def cdf_plot(values):
        """Numpy function to plot CDF."""
        normalized_values = values / np.sum(values)
        sorted_values = np.sort(normalized_values)
        cumulative_values = np.cumsum(sorted_values)
        fraction_of_examples = (
            np.arange(cumulative_values.size, dtype=np.float32) /
            cumulative_values.size)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        ax.plot(fraction_of_examples, cumulative_values)
        ax.set_ylabel('cumulative normalized values')
        ax.set_xlabel('fraction of examples')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(),
                              dtype='uint8').reshape(1, int(height),
                                                     int(width), 3)
        return image

    cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
    tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
    """Adds a tf.summary.image for a histogram plot of the values.

  Plots the histogram of values and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    bins: bin edges which will be directly passed to np.histogram.
    name: name for the image summary.
  """
    def hist_plot(values, bins):
        """Numpy function to plot hist."""
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        y, x = np.histogram(values, bins=bins)
        ax.plot(x[:-1], y)
        ax.set_ylabel('count')
        ax.set_xlabel('value')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(),
                              dtype='uint8').reshape(1, int(height),
                                                     int(width), 3)
        return image

    hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
    tf.summary.image(name, hist_plot)
