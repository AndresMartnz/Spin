import tensorflow as tf


def plot_history_by_key(keys):
    import matplotlib.pyplot as plt

    for key in keys:
        plt.plot(
            history.history[key], marker="o", markersize=0.0, linewidth=1.0, label=key
        )
    plt.xlabel("epoch")
    plt.legend()
    plt.show()
    return history.history[key]


class SaveBestAtEnd(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor="val_loss", mode="min", verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_weights = None
        self.best = float("inf") if mode == "min" else -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if (self.mode == "min" and current < self.best) or (
            self.mode == "max" and current > self.best
        ):
            self.best = current
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.model.save_weights(self.filepath)
            if self.verbose:
                print(
                    f"\nMejores pesos guardados en '{self.filepath}' con {self.monitor} = {self.best:.5f}"
                )
