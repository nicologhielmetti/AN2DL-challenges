import os
import zipfile
from datetime import datetime
import random
import shutil

from CustomDataset import CustomDataset
from NeuralNetworkModel import NeuralNetworkModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image
import json


class NeuralNetworkFlow:
    def __init__(self, seed, dataset_path, n_classes, out_h=256, out_w=256, img_h=256, img_w=256, batch_size=32,
                 n_test_images=15, teams=None, crops=None):
        if crops is None:
            self.crops = ["Haricot", "Mais"]
        if teams is None:
            self.teams = ["Bipbip", "Pead", "Roseau", "Weedelec"]
        self.default_teams = ["Bipbip", "Pead", "Roseau", "Weedelec"]
        self.default_crops = ["Haricot", "Mais"]
        self.out_shape = [out_h, out_w]
        self.seed = seed
        self.dataset_path = dataset_path
        self.n_classes = n_classes
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.n_test_images = n_test_images
        self.models = []
        self.img_generator = None
        self.mask_generator = None
        self.img_data_gen = None
        self.mask_data_gen = None
        self.train_set = None
        self.validation_set = None
        self.train_set_len = None
        self.validation_set_len = None

        self.colors = [cm.rainbow(x) for x in np.linspace(0, 1, 20)]
        random.seed(seed)

    def apply_data_augmentation(self, rotation_range=0, width_shift_range=0, height_shift_range=0,
                                zoom_range=0.5, horizontal_flip=True, vertical_flip=True, fill_mode='reflect'):
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

    @staticmethod
    def list_difference(l1, l2):
        return list(list(set(l1) - set(l2)) + list(set(l2) - set(l1)))

    def _create_splits(self, dataset_path, train_val_split):
        os.chdir(dataset_path)
        train_list = []
        validation_list = []
        final_train_list = []
        final_val_list = []
        for team in self.teams:
            for crop in self.crops:
                temp_tot_list = [os.path.join(os.getcwd(), team, crop, "Images", f) for f in
                                 os.listdir(os.path.join(os.getcwd(), team, crop, "Images")) if
                                 f.endswith('.png') or f.endswith('.jpg')]
                temp_train_list = random.sample(temp_tot_list, int(len(temp_tot_list) * (1 - train_val_split)))
                temp_validation_list = self.list_difference(temp_tot_list, temp_train_list)
            train_list.extend(temp_train_list)
            validation_list.extend(temp_validation_list)

        target_list_train = [e.replace("Images", "Masks")[:-4] + ".png" for e in train_list]
        target_list_validation = [e.replace("Images", "Masks")[:-4] + ".png" for e in validation_list]

        final_train_list.append(train_list)
        final_train_list.append(target_list_train)
        final_val_list.append(validation_list)
        final_val_list.append(target_list_validation)

        if os.path.exists("Splits"):
            shutil.rmtree("Splits")
        os.makedirs("Splits")
        os.chdir("Splits")
        with open("train.json", "w") as train:
            json.dump(final_train_list, train)
        with open("val.json", "w") as validation:
            json.dump(final_val_list, validation)

    def create_train_validation_sets(self, use_data_aug_test_time=False, preprocessing_function=None,
                                     train_val_split=0.3):
        self._create_splits(self.dataset_path, train_val_split)
        train_set = CustomDataset(self.seed, self.dataset_path, 'training', self.img_generator, self.mask_generator,
                                  preprocessing_function, self.out_shape)
        validation_set = CustomDataset(self.seed, self.dataset_path, 'validation',
                                       self.img_generator if use_data_aug_test_time else None,
                                       self.mask_generator if use_data_aug_test_time else None,
                                       preprocessing_function, self.out_shape)
        self.train_set_len = len(train_set)
        self.validation_set_len = len(validation_set)
        self.train_set = tf.data.Dataset.from_generator(lambda: train_set,
                                                        output_types=(tf.float32, tf.float32),
                                                        output_shapes=((*self.out_shape, 3),
                                                                       (*self.out_shape, 1))
                                                        ).batch(self.batch_size).repeat()

        self.validation_set = tf.data.Dataset.from_generator(lambda: validation_set,
                                                             output_types=(tf.float32, tf.float32),
                                                             output_shapes=((*self.out_shape, 3),
                                                                            (*self.out_shape, 1))
                                                             ).batch(self.batch_size).repeat()

    def test_data_generator(self, n_range=10):
        iterator = iter(self.validation_set)

        for _ in range(n_range):
            fig, ax = plt.subplots(1, 2)
            augmented_img, target = next(iterator)
            augmented_img = augmented_img[0]  # First element
            augmented_img = augmented_img  # denormalize - what the hell is this line intended for?!

            target = np.array(target[0, ..., 0])  # First element (squeezing channel dimension)

            print(np.unique(target))

            target_img = np.zeros([*self.out_shape, 3])

            target_img[np.where(target == 0)] = [0, 0, 0]
            for i in range(1, self.n_classes):
                target_img[np.where(target == i)] = np.array(self.colors[i - 1])[:3] * 255

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

    def train_models(self):
        if len(self.models) == 0:
            print("You have to add models before training")
            return
        for model in self.models:
            model.model.fit(x=self.train_set,
                            epochs=model.epochs,
                            steps_per_epoch=self.train_set_len,
                            validation_data=self.validation_set,
                            validation_steps=self.validation_set_len,
                            callbacks=model.callbacks)

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        self.add_neural_network_model(model, compile=False)

    def load_weights(self, weights_path, model, callbacks, epochs, optimizer, loss, metrics):
        model.load_weights(weights_path)
        self.add_neural_network_model(model, callbacks, epochs, optimizer, loss, metrics)

    @staticmethod
    def create_callbacks(experiment_dir_path='exp_dir_chall2', model_name='CNN', save_weights_only=False,
                         early_stopping=True, patience=10, monitor='val_loss'):

        exps_dir = os.path.join('/content/drive/My Drive/', experiment_dir_path)
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
                                                           save_weights_only=save_weights_only, verbose=1,
                                                           save_best_only=True, mode='auto', monitor=monitor)
        callbacks.append(ckpt_callback)
        tb_dir = os.path.join(exp_dir, 'tb_logs')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                     profile_batch=0,
                                                     histogram_freq=1)
        callbacks.append(tb_callback)
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)
            callbacks.append(es_callback)

        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau())
        return callbacks

    def set_neural_network_model(self, model, callbacks=None, epochs=None, optimizer=None, loss=None, metrics=None,
                                 compile=True):
        if len(self.models) > 0:
            self.models.clear()
        self.models.append(NeuralNetworkModel(model, callbacks, epochs, optimizer, loss, metrics, compile))

    def add_neural_network_model(self, model, callbacks=None, epochs=None, optimizer=None, loss=None, metrics=None,
                                 compile=True):
        self.models.append(
            NeuralNetworkModel(model, callbacks, epochs, optimizer, loss, metrics, compile)
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

    # ----- TEST PART

    def _get_list_of_files(self, dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self._get_list_of_files(fullPath)
            elif entry[-4:] == '.png' or entry[-4:] == '.jpg':
                allFiles.append(fullPath)

        return allFiles

    def _rle_encode(self, img):
        ''' img: numpy array, 1 - foreground, 0 - background.Returns run length as string formatted'''
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def test_models(self, test_path):
        default_crops = ["Haricot", "Mais"]
        default_teams = ["Bipbip", "Pead", "Roseau", "Weedelec"]
        test_elements = self._get_list_of_files(test_path)
        test_dataset = list()
        for elem in test_elements:
            img = Image.open(elem)
            img = img.resize([self.img_h, self.img_w], resample=Image.NEAREST)
            img = np.array(img)
            test_dataset.append(img)
        iterator = iter(test_dataset)
        predictions = []
        for model in self.models:
            for _ in range(self.n_test_images):
                image = next(iterator)
                out_sigmoid = model.model.predict(x=tf.expand_dims(image, 0))
                predicted_mask = tf.argmax(out_sigmoid, -1)  # masks
                mask_arr = np.array(predicted_mask)  # converted
                predictions.append(mask_arr)
                fig, ax = plt.subplots(1, 2, figsize=(8, 8))
                fig.show()
                predicted_mask = predicted_mask[0, ...]
                prediction_img = np.zeros([self.img_h, self.img_w, 3])
                prediction_img[np.where(predicted_mask == 0)] = [0, 0, 0]
                for i in range(0, 3):
                    prediction_img[np.where(predicted_mask == i)] = np.array(self.colors[i - 1])[:3] * 255
                ax[0].imshow(np.uint8(image))
                ax[1].imshow(np.uint8(prediction_img))
                fig.canvas.draw()
        submission_dict = {}
        i = 0
        for team in self.teams:
            for crop in self.crops:
                if team in default_teams and crop in default_crops:
                    for el in test_elements:  # create all the keys, this might be redoundant code
                        img_path = el
                        img_name = os.path.basename(img_path)
                        img_name = img_name[:len(img_name) - 4]
                        submission_dict[img_name] = {}
                    for _id in predictions:  # Adding information for keys that matches the value
                        img_path = test_elements[i]
                        i = i + 1
                        img_name = os.path.basename(img_path)
                        img_name = img_name[:len(img_name) - 4]
                        submission_dict[img_name]['shape'] = _id.shape
                        submission_dict[img_name]['team'] = team
                        submission_dict[img_name]['crop'] = crop
                        submission_dict[img_name]['segmentation'] = {}
                        rle_encoded_crop = self._rle_encode(_id == 1)
                        rle_encoded_weed = self._rle_encode(_id == 2)
                        submission_dict[img_name]['segmentation']['crop'] = rle_encoded_crop
                        submission_dict[img_name]['segmentation']['weed'] = rle_encoded_weed

                else:
                    img_list = self._get_list_of_files(test_path + str(team) + '/' + str(crop) + '/Images')
                    for img in img_list:
                        img_name = os.path.basename(img)
                        img_name = img_name[:len(img_name) - 4]
                        submission_dict[img_name] = {}
                        submission_dict[img_name]['shape'] = (2048, 1536)
                        submission_dict[img_name]['team'] = team
                        submission_dict[img_name]['crop'] = crop
                        submission_dict[img_name]['segmentation'] = {}
                        submission_dict[img_name]['segmentation']['crop'] = ""
                        submission_dict[img_name]['segmentation']['weed'] = ""
        print(submission_dict)
        with open('submission.json', 'w') as f:  # dumps the dictionary into the json
            json.dump(submission_dict, f)
        zipfile.ZipFile('submission.zip', mode='w').write("submission.json")

        #from google.colab import files
        #files.download('submission.zip')

    def _rle_decode(self, rle, shape):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    def check_encoding(self, img_name='Bipbip_haricot_im_03691'):
        with open('submission.json', 'r') as f:
            submission_dict = json.load(f)

        img_shape = submission_dict[img_name]['shape']

        rle_encoded_crop = submission_dict[img_name]['segmentation']['crop']
        rle_encoded_weed = submission_dict[img_name]['segmentation']['weed']

        crop_mask = self._rle_decode(rle_encoded_crop, shape=img_shape)
        weed_mask = self._rle_decode(rle_encoded_weed, shape=img_shape)

        reconstructed_mask = crop_mask + (weed_mask * 2)
        reconstructed_rgb_arr = np.zeros(shape=img_shape + [3])
        reconstructed_rgb_arr[reconstructed_mask == 1] = [255, 255, 255]
        reconstructed_rgb_arr[reconstructed_mask == 2] = [216, 67, 82]

        reconstructed_rgb_img = Image.fromarray(
            np.uint8(reconstructed_rgb_arr))

        reconstructed_rgb_img.show()
