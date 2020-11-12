# Check image max size. Useful for data augmentation.
import json
import os
import shutil
from datetime import datetime
from functools import partial

from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard import program

from CNNClassifier import CNNClassifier

SEED = 1996


def divideDatasetInTargetFolders(json_definition, dataset_path):
    for elem in json_definition:
        dest_dir = os.path.join(dataset_path, str(json_definition[elem]))
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        try:
            shutil.move(os.path.join(dataset_path, elem),
                        os.path.join(dest_dir, elem)
                        )
        except FileNotFoundError as e:
            print("File not found: " + str(e))
            continue
    os.mkdir(os.path.join(dataset_path, "augmented"))
    os.mkdir(os.path.join(dataset_path, "augmented/training"))
    os.mkdir(os.path.join(dataset_path, "augmented/validation"))


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
test_path = os.path.join(os.getcwd(), 'MaskDataset/test')

# division_dict = json.load(
#     open(os.path.join(os.getcwd(), 'MaskDataset/train_gt.json'))
# )

# divideDatasetInTargetFolders(division_dict, train_path)


# remember to check both train and test datasets to be sure of max dimensions
max_w, max_h = max(getMaxImageSize(os.path.join(train_path, '0')),
                   getMaxImageSize(os.path.join(train_path, '1')),
                   getMaxImageSize(os.path.join(train_path, '2')))
print("Maximum width and height: " + str((max_w, max_h)))

min_w, min_h = min(getMinImageSize(os.path.join(train_path, '0'), max_w, max_h),
                   getMinImageSize(os.path.join(train_path, '1'), max_w, max_h),
                   getMinImageSize(os.path.join(train_path, '2'), max_w, max_h))
print("Minimum width and height:  " + str((min_w, min_h)))
print("Maximum width  expansion:  " + str(max_w - min_w) + ", increase ratio: " +
      str(float(max_w) / float(max_w - min_w)))
print("Maximum height expansion:  " + str(max_h - min_h) + ", increase ratio: " +
      str(float(max_h) / float(max_h - min_h)))

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

test_data_gen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preproc_fun_fixed)

classes = ['0', '1', '2']
save_dir = os.path.join(train_path, 'augmented')

bs = 8

train_gen = train_data_gen.flow_from_directory(train_path,
                                               target_size=(max_w, max_h),
                                               seed=SEED,
                                               classes=classes,
                                               save_prefix='training_aug',
                                               save_to_dir=os.path.join(save_dir, 'training'),
                                               subset='training',
                                               shuffle=True,
                                               batch_size=bs
                                               )

valid_gen = train_data_gen.flow_from_directory(train_path,
                                               target_size=(max_w, max_h),
                                               seed=SEED,
                                               classes=classes,
                                               save_prefix='validation',
                                               save_to_dir=os.path.join(save_dir, 'validation'),
                                               subset='validation',
                                               shuffle=False,
                                               batch_size=bs
                                               )

test_gen = test_data_gen.flow_from_directory(train_path,
                                             target_size=(max_w, max_h),
                                             seed=SEED,
                                             classes=classes,
                                             shuffle=False,
                                             batch_size=bs
                                             )

train_set = tf.data.Dataset.from_generator(lambda: train_gen,
                                           output_types=(tf.float32, tf.float32),
                                           output_shapes=(
                                               [None, max_w, max_h, 3],
                                               [None, len(classes)]
                                           ))

validation_set = tf.data.Dataset.from_generator(lambda: valid_gen,
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=(
                                                    [None, max_w, max_h, 3],
                                                    [None, len(classes)]
                                                ))

test_set = tf.data.Dataset.from_generator(lambda: test_gen,
                                          output_types=(tf.float32, tf.float32),
                                          output_shapes=(
                                              [None, max_w, max_h, 3],
                                              [None, len(classes)]
                                          ))

train_set.repeat()
validation_set.repeat()
test_set.repeat()

start_f = 8
depth = 3

model = CNNClassifier(depth=depth,
                      start_f=start_f,
                      num_classes=len(classes)
                      )

model.build(input_shape=(None, max_h, max_w, 3))

model.summary()

loss = tf.keras.losses.CategoricalCrossentropy()
# maybe explore learning rate solutions
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

tracking_address = os.path.join(os.getcwd(), "tracking_dir")
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tracking_address])
url = tb.launch()

if not os.path.exists(tracking_address):
    os.makedirs(tracking_address)

now = datetime.now().strftime('%b%d_%H-%M-%S')

model_name = 'CNN'

exp_dir = os.path.join(tracking_address, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

callbacks = []

ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)

# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

early_stop = True
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callbacks.append(es_callback)

model.fit(x=train_set,
          epochs=100,  #### set repeat in training dataset
          steps_per_epoch=len(train_gen),
          validation_data=validation_set,
          validation_steps=len(valid_gen),
          callbacks=callbacks)
