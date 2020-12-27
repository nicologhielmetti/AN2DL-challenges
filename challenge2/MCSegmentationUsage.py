from NeuralNetworkFlow import NeuralNetworkFlow
import tensorflow as tf
import segmentation_models as sm

tf.keras.backend.set_image_data_format('channels_last')

img_w = 256
img_h = 256

models_string = ['resnext101', 'inceptionv3', 'senet154', 'vgg16']

models = []
preproc_fs = []

for model_string in models_string:
    models.append(
        sm.Unet(model_string, classes=3, activation='softmax', input_shape=(img_h, img_w, 3), encoder_weights=None)
    )
    preproc_fs.append(sm.get_preprocessing(model_string))

firstTentative = NeuralNetworkFlow(seed=1996,
                                   dataset_path='/content/Development_Dataset/Training',
                                   n_classes=3,
                                   out_h=img_h, out_w=img_w, img_h=img_h, img_w=img_w,
                                   batch_size=32,
                                   n_test_images=15
                                   )
firstTentative.apply_data_augmentation()
# firstTentative.test_data_generator()

# model = firstTentative.create_custom_model(encoder=tf.keras.applications.VGG16(weights='imagenet', include_top=False,
#                                                                                input_shape=(img_h, img_w, 3)),
#                                            decoder=firstTentative.create_decoder(depth=5, start_filters=32))

for i in range(len(models)):
    firstTentative.create_train_validation_sets(preprocessing_function=preproc_fs[i], use_data_aug_test_time=True)
    firstTentative.set_neural_network_model(models[i], firstTentative.create_callbacks(model_name=models_string[i],
                                                                                       save_weights_only=True,
                                                                                       monitor='val_meanIoU'),
                                            epochs=100,
                                            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                            metrics=[firstTentative.meanIoU, 'accuracy']
                                            )
    firstTentative.train_models()

# firstTentative.load_weights('/content/drive/MyDrive/exp_dir_chall2/FIRST_TENTATIVE_Dec17_00-23-32/ckpts/cp_04.ckpt', model,
#                             firstTentative.create_callbacks(
#                                 model_name="FIRST_TENTATIVE", save_weights_only=True, monitor='val_meanIoU'),
#                             epochs=100, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#                             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                             metrics=['accuracy', firstTentative.meanIoU])
# firstTentative.test_models(test_path='/content/Development_Dataset/Test_Dev')
