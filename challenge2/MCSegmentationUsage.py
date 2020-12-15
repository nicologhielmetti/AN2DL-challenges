from NeuralNetworkFlow import NeuralNetworkFlow
import tensorflow as tf

firstTentative = NeuralNetworkFlow(seed=1996,
                                   dataset_path='/content/Development_Dataset/Training',
                                   n_classes=3,
                                   out_h=256, out_w=256, img_h=2048, img_w=1536,
                                   batch_size=32,
                                   n_test_images=15
                                   )
firstTentative.apply_data_augmentation()
firstTentative.create_train_validation_sets()
for _ in range(10):
    firstTentative.test_data_generator()

model = firstTentative.create_custom_model(encoder=firstTentative.create_encoder(depth=5, start_filters=4),
                                           decoder=firstTentative.create_decoder(depth=5, start_filters=4))

firstTentative.add_neural_network_model(model, firstTentative.create_callbacks(model_name="FIRST_TENTATIVE",
                                                                               save_weights_only=True),
                                        epochs=100,
                                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                        metrics=['accuracy', firstTentative.meanIoU]
                                        )

firstTentative.train_models()
firstTentative.test_models(test_path='/content/Development_Dataset/Test_Dev')