from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
# import funcs

import tensorflow as tf
import numpy as np


class LSTMModel:
    def __init__(self, output_dim, batch_size, no_epochs, no_units,activation='relu'):
        self.output_dim = output_dim
        self.batch_size = batch_size 
        self.no_epochs = no_epochs
        self.no_units = no_units
        self.activation = activation
        self.early_stopping = None
        self.model = None
        self.history = None
        

    # def create_dataset(self, X, window_size=3):
    #     if len(X) < window_size:
    #         raise ValueError("The input array is shorter than the window size.")

    #     if window_size < 1:
    #         raise ValueError("The window size must be at least 1.")
        
    #     Xs, ys = [], []
    #     for i in range(len(X) - window_size):
    #         v = X[i:i + window_size]
    #         Xs.append(v)
    #         ys.append(X[i + window_size])
    #     return np.array(Xs), np.array(ys)


    def generate_model(self,dropout_rate=0.2):
        """ Generate the LSTM model."""
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(self.no_units, return_sequences=True))  
        self.model.add(Dropout(dropout_rate))  # add a Dropout layer to avoid overfitting
        
        # Second LSTM layer
        self.model.add(LSTM(self.no_units, return_sequences=False))  
        self.model.add(Dropout(dropout_rate))  # add a Dropout layer to avoid overfitting
        
        # Fully connected layer
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation(self.activation))


    def generate_and_compile_model(self, patience):
        self.generate_model()  # generate model
        self.early_stopping = EarlyStopping(monitor='val_loss',
                                                               patience=patience,
                                                               mode='min',
                                                               restore_best_weights=True)
        # Compile model
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.legacy.Adam(),
                           metrics=[tf.keras.metrics.MeanSquaredError()])


    def train(self, x_train, y_train, x_val, y_val):
        self.history = self.model.fit(x_train,
                                      y_train,
                                      batch_size=self.batch_size,
                                      epochs=self.no_epochs,
                                      validation_data=(x_val, y_val),
                                      callbacks=[self.early_stopping])


    def predict(self, x_test):
        no_of_pred = x_test.shape[0]
        predictions = self.model.predict(x_test)
        # [-no_of_pred:, -1, :]
        # predictions = np.clip(predictions, 0, 1)
        # predictions = np.round(predictions)
        
        return predictions.reshape(no_of_pred, self.output_dim)


    def validate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", scores[0])
        print("Test accuracy:", scores[1])


    def save(self):
        self.model.save('lstm.keras')


    def plot_loss(self):

        """
        This function plots the loss and validation loss of a trained TensorFlow model as a function of epochs.

        Parameters:
        history (tensorflow.python.keras.callbacks.History): The history object returned from the fit method of a TensorFlow model.

        Returns:
        None

        Note:
        The function plots the 'loss' and 'val_loss' from the history object against the epoch number.
        'loss' is represented by a blue line ('Loss') and 'val_loss' is represented by an orange line ('Val_loss').
        It will show the plot using 'plt.show()'. Be sure that your environment supports plot visualization.
        """
        
        loss = (self.history.history['loss'])
        val_loss = (self.history.history['val_loss'])

        epochs = range(1, len(loss) + 1)

        plt.figure()
        plt.semilogy(epochs, loss, label='Training loss')  # 'bo' stands for 'blue dot'
        plt.semilogy(epochs, val_loss, label='Validation loss')  # 'b' stands for 'solid blue line'
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/log scale')
        plt.legend()
        plt.show()