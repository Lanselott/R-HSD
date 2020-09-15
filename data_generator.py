import hashlib
import __future__
import sys
sys.path.insert(0, '/research/byu2/rchen/models/research/')
import os
import io
import numpy as np

#import gdspy
import csv
import cv2
import tensorflow as tf

from lxml import etree
from PIL import Image
from IPython import embed
from skimage.io import imread

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

Image.MAX_IMAGE_PIXELS = 1230046000  # load large image

def data_generator(index, state, cf_data_num_h, writer, raw_in_data, label_in_data):
    data_path = './benchmark/temp/'
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
    '''determine how many hotspots in the cropped image can be labeled as class=1 '''
    threshold = 255

    while cf_data_num_h > 0:  # or cf_data_num_nh >= 0:
        if index == 4:
            cf_crop_size = 128

        # NOTE: we use left half as training data
        random_x = abs(np.random.random_integers(0, width / 2) - cf_crop_size)
        random_y = abs(np.random.random_integers(0, height) - cf_crop_size)

        bottom_right_x = random_x + cf_crop_size
        bottom_right_y = random_y + cf_crop_size

        crop_data = raw_im[random_y:bottom_right_y, random_x:bottom_right_x]
        crop_label = label_im[random_y:bottom_right_y, random_x:bottom_right_x]
        sum_label = int(cv2.sumElems(crop_label)[0])
        '''
        crop the label 4 with small size and upsample to 128 -> 256
        * INTER_NEAREST, make sure the hotspot pixel value is 255
        '''
        if index == 4:
            cf_crop_size = 256
            crop_data = cv2.resize(crop_data, (cf_crop_size, cf_crop_size),
                                   interpolation=cv2.INTER_NEAREST)
            crop_label = cv2.resize(crop_label, (cf_crop_size, cf_crop_size),
                                    interpolation=cv2.INTER_NEAREST)

        ismiddle = 0  # check whether in the middle or not
        delta = 34
        mask_type = 'png'

        if index == 4:
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

        if sum_label != 0 and sum_label >= threshold and cf_data_num_h > 0:
            # find hotspot
            detected = 1
            h, w = crop_data.shape

            if h == cf_crop_size and w == cf_crop_size:
                cv2.imwrite(data_path + str(index) + '_' + str(count) + '.png',
                            crop_data)
                cv2.imwrite(
                    data_path + str(index) + '_' + str(count) + '_label.png',
                    crop_label)
                '''generate bounding box'''
                for j in range(cf_crop_size - 4):
                    for i in range(cf_crop_size - 4):
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

                if ismiddle:
                    img_path = data_path + \
                        str(index) + '_' + str(count) + '.png'
                    label_path = data_path + \
                        str(index) + '_' + str(count) + '_label.png'
                    mask_path = data_path + \
                        str(index) + '_' + str(count) + '_mask.png'
                        
                    cv2.imwrite(img_path, crop_data)
                    cv2.imwrite(label_path, crop_label)
                    cv2.imwrite(mask_path, crop_label)

                    filename = str(index) + '_' + str(count) + '.png'
                    count += 1
                    cf_data_num_h -= 1
                    with tf.gfile.GFile(img_path, 'rb') as fid:
                        encoded_png = fid.read()
                    encoded_png_io = io.BytesIO(encoded_png)
                    image = Image.open(encoded_png_io)
                    if image.format != 'PNG':
                        raise ValueError('Image format not PNG')
                    key = hashlib.sha256(encoded_png).hexdigest()

                    with tf.gfile.GFile(mask_path, 'rb') as fid:
                        encoded_mask_png = fid.read()
                    encoded_png_mask_io = io.BytesIO(encoded_mask_png)
                    mask = Image.open(encoded_png_mask_io)
                    if mask.format != 'PNG':
                        raise ValueError('Mask format not PNG')

                    mask_np = np.asarray(mask)
                    mask_remapped = (mask_np != 2).astype(np.uint8)
                    masks.append(mask_remapped)

                    encoded_mask_png_list = []
                    for mask in masks:
                        img = Image.fromarray(mask)
                        output = io.BytesIO()
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
    state = 'train'
    merged_samples = True
    sample_num = 3000

    if merged_samples:
        iccad_benchmark_list = [2]
        tfrecord_name = 'hsd_od_merged_' + state + '.record'
        raw_in_data = os.path.join('./benchmark/raw_merged' + '.png')
        label_in_data = os.path.join('./benchmark/label_merged' + '.png')
    else:
        iccad_benchmark_list = [2, 3, 4]
        tfrecord_name = 'hsd_od_' + state + '.record'

    writer = tf.python_io.TFRecordWriter(
        '/research/byu2/rchen/proj/cuhsd/hsd-od/benchmark/tfrecord/' + tfrecord_name)

    for iccad_index in iccad_benchmark_list:
        if not merged_samples:
            raw_in_data = os.path.join('./benchmark/raw' + str(iccad_index) + '.png')
            label_in_data = os.path.join('./benchmark/label' + str(iccad_index) + '.png')
        
        data_generator(iccad_index, state, sample_num, writer, raw_in_data, label_in_data)
        print("iccad benchmark {} done.".format(iccad_index))

    writer.close()
