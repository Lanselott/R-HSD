import os
import math
import numpy as np
import cv2

from random import shuffle
from IPython import embed

SAMPLE_SIZE = 256
CUT_SIZE = 137 #* 25

def labeling(file):
    label_sample = np.zeros([SAMPLE_SIZE, SAMPLE_SIZE])
    
    if file[:2] == "HS":
        label_sample[SAMPLE_SIZE//2, SAMPLE_SIZE//2] = 255
    
    return label_sample

def merge_clip(path, train):
    if train:
        path += 'train'
    else:
        path += 'test'

    
    sample_nums = 5

    for root, dirs, files in os.walk(path):
        shuffle(files)

        num_samples = len(files) # 18739
        split_list = [cut for cut in range(0, 18770, CUT_SIZE * CUT_SIZE)]
        # split_list.append(18739)

        for j in range(len(split_list) - 1):
            subfiles = files[split_list[j]:split_list[j + 1]]
            image_merged = np.zeros([SAMPLE_SIZE * CUT_SIZE, SAMPLE_SIZE * CUT_SIZE])
            label_merged = np.zeros([SAMPLE_SIZE * CUT_SIZE, SAMPLE_SIZE* CUT_SIZE])

            for i in range(len(subfiles)):
                file = subfiles[i]
                hs_sample = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                hs_sample = cv2.resize(hs_sample, (SAMPLE_SIZE, SAMPLE_SIZE))
                label_sample = labeling(file)
                x = i % CUT_SIZE
                y = i / CUT_SIZE
                image_merged[x*SAMPLE_SIZE : (x+1)*SAMPLE_SIZE, y*SAMPLE_SIZE : (y+1)*SAMPLE_SIZE] = hs_sample
                label_merged[x*SAMPLE_SIZE : (x+1)*SAMPLE_SIZE, y*SAMPLE_SIZE : (y+1)*SAMPLE_SIZE] = label_sample

            cv2.imwrite(str(split_list[j + 1]) + "image_merged.png", image_merged)
            cv2.imwrite(str(split_list[j + 1]) + "label_merged.png", label_merged)

if __name__ == '__main__':
    iccad12_path = './iccad-official/'
    train = True
    
    merge_clip(iccad12_path, train)