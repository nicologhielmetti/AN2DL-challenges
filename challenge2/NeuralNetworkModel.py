class NeuralNetworkModel:
    def __init__(self, model, callbacks, epochs, optimizer, loss, metrics, compile):
        self.model = model
        self.callbacks = callbacks
        self.epochs = epochs

        if compile:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
