# Check image max size. Useful for data augmentation.
import json
import os
import shutil
from functools import partial

from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 1996


def divideDatasetInTargetFolders(json_definition, dataset_path):
    for elem in json_definition:
        dest_dir =  os.path.join(dataset_path, str(json_definition[elem]))
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        shutil.move(os.path.join(dataset_path, elem),
                    os.path.join(dest_dir, elem)
                    )
    aug_path = os.path.join(train_path, 'augmented')
    if not os.path.isdir(aug_path):
        os.mkdir(aug_path)


def getMaxImageSize(dataset_dir):
    max_w = 0
    max_h = 0
    path = os.path.join(os.getcwd(), dataset_dir)
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image = Image.open(os.path.join(path, filename))
            width, height = image.size
            max_w = width if width > max_w else max_w
            max_h = height if height > max_h else max_h
        else:
            print("This file -> " + filename + " is not .jpg")
    return max_w, max_h


def getMinImageSize(dataset_dir, max_w, max_h):
    min_w = max_w
    min_h = max_h
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            image = Image.open(os.path.join(dataset_dir, filename))
            width, height = image.size
            min_w = width if width < min_w else min_w
            min_h = height if height < min_h else min_h
        else:
            print("This file -> " + filename + " is not .jpg")
    return min_w, min_h


train_path = os.path.join(os.getcwd(), 'MaskDataset/training')
test_path  = os.path.join(os.getcwd(), 'MaskDataset/test')

# division_dict = json.load(
#    open(os.path.join(os.getcwd(), 'MaskDataset/train_gt.json'))
#         )
# divideDatasetInTargetFolders(division_dict, train_path)

# remember to check both train and test datasets to be sure of max dimensions
max_w, max_h = max(getMaxImageSize(os.path.join(train_path, '0')),
                   getMaxImageSize(os.path.join(train_path, '1')),
                   getMaxImageSize(os.path.join(train_path, '2')))
print("Maximum width and height: " + str((max_w, max_h)))

min_w, min_h = min(getMinImageSize(os.path.join(train_path, '0'), max_w, max_h),
                   getMinImageSize(os.path.join(train_path, '1'), max_w, max_h),
                   getMinImageSize(os.path.join(train_path, '2'), max_w, max_h))
print("Minimum width and height:  " + str((min_w,  min_h)))
print("Maximum width  expansion:  " + str(max_w - min_w) + ", increase ratio: " +
      str(float(max_w)/float(max_w - min_w)))
print("Maximum height expansion:  " + str(max_h - min_h) + ", increase ratio: " +
      str(float(max_h)/float(max_h - min_h)))

# interpolate param of the following function should be explored
preproc_fun_fixed = partial(tf.keras.preprocessing.image.smart_resize, size=(max_w, max_h))

train_data_gen = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=10,
                                    height_shift_range=10,
                                    zoom_range=0.3,
                                    horizontal_flip=True,
                                    fill_mode='reflect',
                                    rescale=1. / 255,
                                    validation_split=0.3,
                                    preprocessing_function=preproc_fun_fixed
                                    )

classes = ['0', '1', '2']
save_dir = os.path.join(train_path, 'augmented')

train_gen = train_data_gen.flow_from_directory(train_path,
                                   target_size=(max_w, max_h),
                                   seed=SEED,
                                   classes=classes,
                                   save_prefix='augmented_',
                                   save_to_dir=os.path.join(save_dir, 'training'),
                                   subset='training'
                                   )

valid_gen = train_data_gen.flow_from_directory(train_path,
                                   target_size=(max_w, max_h),
                                   seed=SEED,
                                   classes=classes,
                                   save_prefix='augmented_',
                                   save_to_dir=os.path.join(save_dir, '/validation'),
                                   subset='validation'
                                   )

train_set = tf.data.Dataset.from_generator(train_gen,
                                           )
