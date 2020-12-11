import multiprocessing
import os
from datetime import datetime
from CustomDataset import CustomDataset
from NeuralNetworkModel import NeuralNetworkModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


class NeuralNetworkFlow:
    def __init__(self, seed, dataset_dir,  n_classes, img_generator=None, mask_generator=None, out_h=256, out_w=256,
                 img_h=256, img_w=256, batch_size=32):

        self.out_shape = [out_h, out_w]
        self.seed = seed
        self.dataset_dir = dataset_dir
        self.img_generator = img_generator
        self.mask_generator = mask_generator
        self.n_classes = n_classes
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.models = []
        self.img_data_gen = None
        self.mask_data_gen = None
        self.train_set = None
        self.validation_set = None
        self.train_set_len = None
        self.validation_set_len = None

    def apply_data_augmentation(self, rotation_range=10, width_shift_range=10, height_shift_range=10,
                                zoom_range=0.3, horizontal_flip=True, vertical_flip=True, fill_mode='reflect'):
        self.img_data_gen = ImageDataGenerator(rotation_range=rotation_range,
                                               width_shift_range=width_shift_range,
                                               height_shift_range=height_shift_range,
                                               zoom_range=zoom_range,
                                               horizontal_flip=horizontal_flip,
                                               vertical_flip=vertical_flip,
                                               fill_mode=fill_mode)
        self.mask_data_gen = ImageDataGenerator(rotation_range=rotation_range,
                                                width_shift_range=width_shift_range,
                                                height_shift_range=height_shift_range,
                                                zoom_range=zoom_range,
                                                horizontal_flip=horizontal_flip,
                                                vertical_flip=vertical_flip,
                                                fill_mode=fill_mode)

    def create_train_validation_sets(self, train_path, validation_path, use_data_aug_test_time=False,
                                     preprocessing_function=None):
        train_set = CustomDataset(self.seed, train_path, 'training', self.img_generator, self.mask_generator,
                                  preprocessing_function, self.out_shape)
        validation_set = CustomDataset(self.seed, validation_path, 'validation',
                                       self.img_generator if use_data_aug_test_time else None,
                                       self.mask_generator if use_data_aug_test_time else None,
                                       preprocessing_function, self.out_shape)
        self.train_set_len = len(train_set)
        self.validation_set_len = len(validation_set)
        self.train_set = tf.data.Dataset.from_generator(lambda: train_set,
                                                        output_types=(tf.float32, tf.float32),
                                                        output_shapes=([*self.out_shape, 3],
                                                                       [*self.out_shape, 1])
                                                        ).batch(self.batch_size).repeat()

        self.validation_set = tf.data.Dataset.from_generator(lambda: validation_set,
                                                             output_types=(tf.float32, tf.float32),
                                                             output_shapes=([*self.out_shape, 3],
                                                                            [*self.out_shape, 1])
                                                             ).batch(self.batch_size).repeat()
        # @TODO: read_rgb_mask

    def test_data_generator(self):
        evenly_spaced_interval = np.linspace(0, 1, 20)
        colors = [cm.rainbow(x) for x in evenly_spaced_interval]
        iterator = iter(self.validation_set)
        fig, ax = plt.subplots(1, 2)

        augmented_img, target = next(iterator)
        augmented_img = augmented_img[0]  # First element
        augmented_img = augmented_img  # denormalize

        target = np.array(target[0, ..., 0])  # First element (squeezing channel dimension)

        print(np.unique(target))

        target_img = np.zeros([target.shape[0], target.shape[1], 3])

        target_img[np.where(target == 0)] = [0, 0, 0]
        for i in range(1, 21):
            target_img[np.where(target == i)] = np.array(colors[i - 1])[:3] * 255

        ax[0].imshow(np.uint8(augmented_img))
        ax[1].imshow(np.uint8(target_img))

        plt.show()

    @staticmethod
    def create_custom_model(encoder, decoder):
        model = tf.keras.Sequential()
        model.add(encoder)
        model.add(decoder)
        return model

    def create_decoder(self, depth, start_filters, interpolation='bilinear'):
        decoder = tf.keras.Sequential()
        for i in range(depth):
            decoder.add(tf.keras.layers.UpSampling2D(2, interpolation=interpolation))
            decoder.add(tf.keras.layers.Conv2D(filters=start_filters,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding='same'))
            decoder.add(tf.keras.layers.ReLU())
            start_filters = start_filters // 2
        decoder.add(tf.keras.layers.Conv2D(filters=self.n_classes,
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding='same',
                                           activation='softmax'))
        return decoder

    @staticmethod
    def create_encoder(depth, start_filters):
        encoder = tf.keras.Sequential()
        for i in range(depth):
            encoder.add(tf.keras.layers.Conv2D(filters=start_filters,
                                               kernel_size=(3, 3),
                                               strides=(1, 1),
                                               padding='same'))
            encoder.add(tf.keras.layers.ReLU())
            encoder.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
            start_filters *= 2
        return encoder

    def _fit_single_model(self, single_model):
        return single_model.model.fit(x=self.train_set,
                                      epochs=single_model.epochs,
                                      steps_per_epoch=self.train_set_len,
                                      validation_data=self.validation_set,
                                      validation_steps=self.validation_set_len,
                                      callbacks=single_model.callbacks)

    def train_models(self):
        if len(self.models) == 0:
            print("You have to add models before training")
            return
        pool = multiprocessing.Pool(processes=None)
        self.models = pool.map(self._fit_single_model, self.models)
        pool.close()
        pool.join()

    @staticmethod
    def create_callbacks(experiment_dir_path='exp_dir_chall2', model_name='CNN', save_weights_only=False,
                         early_stopping=True, patience=10):
        cwd = os.getcwd()
        exps_dir = os.path.join(cwd, 'drive/My Drive/', experiment_dir_path)
        if not os.path.exists(exps_dir):
            os.makedirs(exps_dir)
        now = datetime.now().strftime('%b%d_%H-%M-%S')
        exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        callbacks = []
        ckpt_dir = os.path.join(exp_dir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'),
                                                           save_weights_only=save_weights_only)
        callbacks.append(ckpt_callback)
        tb_dir = os.path.join(exp_dir, 'tb_logs')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                     profile_batch=0,
                                                     histogram_freq=1)  # if 1 shows weights histograms
        callbacks.append(tb_callback)
        if early_stopping:
            es_callback = tf.keras.callback.EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(es_callback)
        return callbacks

    def add_neural_network_model(self, model, callbacks, epochs, optimizer, loss, metrics):
        self.models.append(
            NeuralNetworkModel(model, callbacks, epochs, optimizer, loss, metrics)
        )

    def meanIoU(self, y_true, y_pred):
        y_pred = tf.expand_dims(tf.argmax(y_pred, -1), -1)

        per_class_iou = []

        for i in range(1, self.n_classes):  # exclude the background class 0
            # Get prediction and target related to only a single class (i)
            class_pred = tf.cast(tf.where(y_pred == i, 1, 0), tf.float32)
            class_true = tf.cast(tf.where(y_true == i, 1, 0), tf.float32)
            intersection = tf.reduce_sum(class_true * class_pred)
            union = tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) - intersection

            iou = (intersection + 1e-7) / (union + 1e-7)
            per_class_iou.append(iou)

        return tf.reduce_mean(per_class_iou)

