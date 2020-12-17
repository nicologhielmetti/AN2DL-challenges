from NeuralNetworkFlow import NeuralNetworkFlow
import tensorflow as tf
import segmentation_models as sm

img_h = 2048
img_w = 1536

model = sm.Unet('resnet101', classes=3, activation='softmax')
preproc_f = sm.get_preprocessing('resnet101')

firstTentative = NeuralNetworkFlow(seed=1996,
                                   dataset_path='/content/Development_Dataset/Training',
                                   n_classes=3,
                                   out_h=256, out_w=256, img_h=img_h, img_w=img_w,
                                   batch_size=32,
                                   n_test_images=15
                                   )
firstTentative.apply_data_augmentation()
firstTentative.create_train_validation_sets(preprocessing_function=preproc_f, use_data_aug_test_time=True)
# firstTentative.test_data_generator()

# model = firstTentative.create_custom_model(encoder=tf.keras.applications.VGG16(weights='imagenet', include_top=False,
#                                                                                input_shape=(img_h, img_w, 3)),
#                                            decoder=firstTentative.create_decoder(depth=5, start_filters=32))

# firstTentative.add_neural_network_model(model, firstTentative.create_callbacks(model_name="FIRST_TENTATIVE",
#                                                                                save_weights_only=True),
#                                         epochs=100,
#                                         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#                                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                                         metrics=['accuracy', firstTentative.meanIoU]
#                                         )

# firstTentative.train_models()
firstTentative.load_weights('/content/MyDrive/exp_dir_chall2/FIRST_TENTATIVE_Dec17_00-23-32/ckpts/cp_04.ckpt', model,
                            firstTentative.create_callbacks(model_name="FIRST_TENTATIVE", save_weights_only=True),
                            epochs=100, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy', firstTentative.meanIoU])
firstTentative.test_models(test_path='/content/Development_Dataset/Test_Dev')
