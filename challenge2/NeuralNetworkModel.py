class NeuralNetworkModel:
    def __init__(self, model, callbacks, epochs, optimizer, loss, metrics):
        self.model = model
        self.callbacks = callbacks
        self.epochs = epochs

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
