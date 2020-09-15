import __future__
import sys
import os
import io
# sys.path.insert(
#     0,
#     '/research/byu2/rchen/proj/region_based_hotspot_detection/detection_models/research'
# )
import numpy as np
import csv
import cv2

import PIL.Image
import hashlib
import tensorflow as tf

from PIL import Image
from skimage.io import imread
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from IPython import embed

Image.MAX_IMAGE_PIXELS = 9613410400  # 1230046000  # load large image


def data_generator(index,
                   state,
                   writer,
                   raw_in_data,
                   label_in_data,
                   data_path,
                   benchmark_name,
                   apply_overlap=True,
                   subset_nums=None):
    '''read as gray'''
    raw_im = imread(raw_in_data, 0)
    label_im = imread(label_in_data, 0)
    '''get whole image info'''
    height, width = label_im.shape  # raw_im.shape

    print("Raw image info: ", height, width)
    '''crop size & num of generate image'''
    cf_crop_size = 256
    detected = 0
    count = 0
    hotspot_num = 0
    x_overlap = 0
    y_overlap = 0
    '''determine how many hotspots in the cropped image can be labeled as positive sample '''
    threshold = 255
    ordered_x = 0
    ordered_y = 0

    if apply_overlap:
        offset = 46
    else:
        offset = 0

    move_step = 256 - offset

    if index == 4:
        move_step = 128
    if benchmark_name is 'iccad16':
        width = width / 2

    #NOTE: no random in eval part, crop one by one
    for k in range(height / move_step + 1):
        for l in range(width / move_step + 1):
            if count == subset_nums:
                break
            '''iccad 16, when index = 4, 256->128'''
            if index == 4:
                cf_crop_size = 128
            if benchmark_name is 'iccad16':
                ordered_x = width / 2 + l * move_step
            else:
                ordered_x = l * move_step

            ordered_y = k * move_step
            bottom_right_x = ordered_x + cf_crop_size
            bottom_right_y = ordered_y + cf_crop_size

            crop_data = raw_im[ordered_y:bottom_right_y, ordered_x:
                               bottom_right_x]
            crop_label = label_im[ordered_y:bottom_right_y, ordered_x:
                                  bottom_right_x]
            #NOTE: check bottom_right corner cross border or not
            #UPDATE: 11/20, we fill the blank on the overlap part
            if bottom_right_x > width:
                x_overlap = bottom_right_x - width
                crop_data = np.pad(crop_data, ((0, 0), (0, x_overlap)),
                                   mode='constant',
                                   constant_values=0)
                crop_label = np.pad(crop_label, ((0, 0), (0, x_overlap)),
                                    mode='constant',
                                    constant_values=0)

            if bottom_right_y > height:
                y_overlap = bottom_right_y - height
                crop_data = np.pad(crop_data, ((0, y_overlap), (0, 0)),
                                   mode='constant',
                                   constant_values=0)
                crop_label = np.pad(crop_label, ((0, y_overlap), (0, 0)),
                                    mode='constant',
                                    constant_values=0)

            sum_label = int(cv2.sumElems(crop_label)[0])
            '''
            *****  crop the label 4 with small size and upsample to 128 -> 256*****
            INTER_NEAREST, make sure the hotspot pixel value is 255
            '''
            if index == 4:
                cf_crop_size = 256
                crop_data = cv2.resize(crop_data, (cf_crop_size, cf_crop_size),
                                       interpolation=cv2.INTER_NEAREST)
                crop_label = cv2.resize(crop_label,
                                        (cf_crop_size, cf_crop_size),
                                        interpolation=cv2.INTER_NEAREST)

            ismiddle = 0  # check whether in the middle or not
            delta = 34
            mask_type = 'png'
            if index == 4 and benchmark_name == 'iccad16':
                delta = 68
            object_id = index - 1
            object_name = 'hotspot_' + str(object_id)

            xmin = []
            ymin = []
            xmax = []
            ymax = []
            classes = []
            classes_text = []
            masks = []
            h, w = crop_data.shape

            if h == cf_crop_size and w == cf_crop_size:
                # cv2.imwrite(data_path + str(index) + '_' + str(count) + '.png',
                #             crop_data)
                # cv2.imwrite(
                #     data_path + str(index) + '_' + str(count) + '_label.png',
                #     crop_label)
                '''generate bounding box'''
                for j in range(cf_crop_size - 4):
                    for i in range(cf_crop_size - 4):
                        # and crop_label[j+4, i+4] == 255:
                        if (index != 4 and crop_label[j, i] == 255) or (
                                index == 4 and np.sum(
                                    crop_label[j:j + 256 / 128, i:i +
                                               256 / 128]) == 2 * 2 * 255):
                            ismiddle = 1
                            box_min_x = i - delta
                            box_min_y = j - delta
                            box_max_x = i + delta
                            box_max_y = j + delta

                            if box_min_x < 0:
                                box_min_x = 1

                            if box_min_y < 0:
                                box_min_y = 1

                            if box_max_x >= cf_crop_size:
                                box_max_x = cf_crop_size - 1

                            if box_max_y >= cf_crop_size:
                                box_max_y = cf_crop_size - 1

                            xmin.append(float(box_min_x) / float(cf_crop_size))
                            ymin.append(float(box_min_y) / float(cf_crop_size))
                            xmax.append(float(box_max_x) / float(cf_crop_size))
                            ymax.append(float(box_max_y) / float(cf_crop_size))
                            classes_text.append(object_name.encode('utf8'))
                            classes.append(object_id)
                            hotspot_num += 1
                # duplicate condition
                if True:  #ismiddle == 1:
                    full_path = data_path + str(index) + '_' + str(
                        count) + '.png'
                    mask_path = data_path + str(index) + '_' + str(
                        count) + '_mask.png'
                    cv2.imwrite(full_path, crop_data)
                    cv2.imwrite(mask_path, crop_label)

                    filename = str(index) + '_' + str(count) + '.png'
                    count += 1
                    with tf.gfile.GFile(full_path, 'rb') as fid:
                        encoded_png = fid.read()
                    encoded_png_io = io.BytesIO(encoded_png)
                    image = PIL.Image.open(encoded_png_io)
                    if image.format != 'PNG':
                        raise ValueError('Image format not PNG')
                    key = hashlib.sha256(encoded_png).hexdigest()

                    with tf.gfile.GFile(mask_path, 'rb') as fid:
                        encoded_mask_png = fid.read()
                    encoded_png_mask_io = io.BytesIO(encoded_mask_png)
                    mask = PIL.Image.open(encoded_png_mask_io)
                    if mask.format != 'PNG':
                        raise ValueError('Mask format not PNG')

                    mask_np = np.asarray(mask)
                    mask_remapped = (mask_np != 2).astype(np.uint8)
                    masks.append(mask_remapped)

                    encoded_mask_png_list = []
                    for mask in masks:
                        img = PIL.Image.fromarray(mask)
                        output = io.BytesIO()
                        #print("output:",output)
                        img.save(output, format='PNG')
                        encoded_mask_png_list.append(output.getvalue())

                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image/height':
                            dataset_util.int64_feature(h),
                            'image/width':
                            dataset_util.int64_feature(w),
                            'image/filename':
                            dataset_util.bytes_feature(filename.encode(
                                'utf8')),
                            'image/source_id':
                            dataset_util.bytes_feature(filename.encode(
                                'utf8')),
                            'image/key/sha256':
                            dataset_util.bytes_feature(key.encode('utf8')),
                            'image/encoded':
                            dataset_util.bytes_feature(encoded_png),
                            'image/format':
                            dataset_util.bytes_feature('png'.encode('utf8')),
                            'image/object/bbox/xmin':
                            dataset_util.float_list_feature(xmin),
                            'image/object/bbox/xmax':
                            dataset_util.float_list_feature(xmax),
                            'image/object/bbox/ymin':
                            dataset_util.float_list_feature(ymin),
                            'image/object/bbox/ymax':
                            dataset_util.float_list_feature(ymax),
                            'image/object/class/text':
                            dataset_util.bytes_list_feature(classes_text),
                            'image/object/class/label':
                            dataset_util.int64_list_feature(classes),
                        }))

                    writer.write(example.SerializeToString())


if __name__ == '__main__':
    state = 'eval'
    apply_overlap = False  # True
    merged_samples = True
    subset_nums = 3000
    benchmark_name = 'iccad12'  #'iccad16'
    data_path = './benchmark/temp/'

    if merged_samples:
        iccad_benchmark_list = [2]
        if apply_overlap:
            tfrecord_name = 'hsd_od_merged_' + state + '.record'
            data_path += 'hsd_od_merged_' + state + '/'
        else:
            tfrecord_name = 'hsd_od_merged_single_no_overlap_' + state + '.record'
            data_path += 'hsd_od_merged_single_no_overlap_' + state + '/'

        raw_in_data = os.path.join('./benchmark/' + state + '_raw_merged' +
                                   '.png')
        label_in_data = os.path.join('./benchmark/' + state + '_label_merged' +
                                     '.png')
    else:
        iccad_benchmark_list = [2, 3, 4]
        tfrecord_name = 'hsd_od_' + state + '.record'

    writer = tf.python_io.TFRecordWriter(
        '/research/byu2/rchen/proj/cuhsd/hsd-od/benchmark/tfrecord/' +
        tfrecord_name)

    for iccad_index in iccad_benchmark_list:
        if not merged_samples:
            raw_in_data = os.path.join('./benchmark/raw' + str(iccad_index) +
                                       '.png')
            label_in_data = os.path.join('./benchmark/label' +
                                         str(iccad_index) + '.png')

        data_generator(iccad_index, state, writer, raw_in_data, label_in_data,
                       data_path, benchmark_name, apply_overlap, subset_nums)
        print("iccad benchmark {} done.".format(iccad_index))

    writer.close()