# Keras API 
from tensorflow import keras

# Deep learning 
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Concatenate, SimpleRNN, Masking, Flatten
from keras import losses
import numpy as np

# Defining hyper parameters 
n_layer = 60
batch = 16
epochs = 200
lr = 0.0001 

# Defining the model class that uses auxilary data for seasonal effects
class NNmodelAux():
    
    def __init__(
        self, 
        X, 
        Xaux,
        Y, 
        n_lag, 
        n_ft,
        n_layer=n_layer,
        batch=batch,
        epochs=epochs, 
        lr=lr,
        Xval=None,
        Xauxval=None,
        Yval=None
    ):
        lstm_input = Input(shape=(n_lag, n_ft))
        dummy_input = Input(shape=(Xaux.shape[1],))

        # Series signal 
        lstm_layer = LSTM(n_layer, activation='relu')(lstm_input)

        # Concatenating 
        x = Concatenate(axis=1)([lstm_layer, dummy_input])
        x = Dense(1)(x)
        
        self.model = Model(inputs=[lstm_input, dummy_input], outputs=x)
        self.batch = batch 
        self.epochs = epochs
        self.lr = lr 
        self.Xval = Xval
        self.Xauxval = Xauxval
        self.Yval = Yval
        self.X = X
        self.Xaux = Xaux
        self.Y = Y
        
    def train(self):
        # Getting the untrained model 
        empty_model = self.model
        
        # Initiating the optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model
        empty_model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizer)

        if (self.Xval is not None) & (self.Yval is not None) & (self.Xauxval is not None):
            history = empty_model.fit(
                [self.X, self.Xaux], 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch, 
                validation_data=([self.Xval, self.Xauxval], self.Yval), 
                shuffle=False
            )
        else:
            history = empty_model.fit(
                [self.X, self.Xaux], 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch,
                shuffle=False
            )
        
        # Saving to original model attribute in the class
        self.model = empty_model
        
        # Returning the training history
        return history
    
    def predict(self, X):
        return self.model.predict(X)


class NNmodel():
    
    def __init__(
        self, 
        X, 
        Y, 
        n_lag, 
        n_ft,
        n_layer=n_layer,
        batch=batch,
        epochs=epochs, 
        lr=lr,
        Xval=None,
        Yval=None
    ):
        lstm_input = Input(shape=(n_lag, n_ft))

        # Series signal 
        lstm_layer = LSTM(n_layer, activation='relu')(lstm_input)

        x = Dense(1)(lstm_layer)
        
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch 
        self.epochs = epochs
        self.lr = lr 
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        
    def train(self):
        # Getting the untrained model 
        empty_model = self.model
        
        # Initiating the optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model
        empty_model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizer)

        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch, 
                validation_data=(self.Xval, self.Yval), 
                shuffle=False
            )
        else:
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch,
                shuffle=False
            )
        
        # Saving to original model attribute in the class
        self.model = empty_model
        
        # Returning the training history
        return history
    
    def predict(self, X):
        return self.model.predict(X)


class NNMultistepModel():
    
    def __init__(
        self, 
        X, 
        Y, 
        n_outputs,
        n_lag,
        n_ft,
        n_layer=n_layer,
        batch=batch,
        epochs=epochs, 
        lr=lr,
        Xval=None,
        Yval=None
    ):
        lstm_input = Input(shape=(n_lag, n_ft))

        # Masking layer 
        lstm_layer = Masking(mask_value=0)(lstm_input)

        # Series signal 
        lstm_layer = LSTM(n_layer, activation='relu')(lstm_layer)

        x = Dense(n_outputs)(lstm_layer)
        
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch 
        self.epochs = epochs
        self.n_layer=n_layer
        self.lr = lr 
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        
    def train(self):
        # Getting the untrained model 
        empty_model = self.model
        
        # Initiating the optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model
        empty_model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizer)

        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch, 
                validation_data=(self.Xval, self.Yval), 
                shuffle=False
            )
        else:
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch,
                shuffle=False
            )
        
        # Saving to original model attribute in the class
        self.model = empty_model
        
        # Returning the training history
        return history
    
    def predict(self, X):
        return self.model.predict(X)


class NNMultiCountryModel():
    
    def __init__(
        self, 
        X, 
        Y, 
        n_outputs,
        n_lag,
        n_ft,
        n_layer=n_layer,
        batch=batch,
        epochs=epochs, 
        lr=lr,
        Xval=None,
        Yval=None
    ):
        # Initiating a model for 3 countries
        lstm1 = Input(shape=(n_lag, n_ft))
        lstm2 = Input(shape=(n_lag, n_ft))
        lstm3 = Input(shape=(n_lag, n_ft))

        # Masking layer
        layer1 = Masking(mask_value=0)(lstm1)
        layer2 = Masking(mask_value=0)(lstm2)
        layer3 = Masking(mask_value=0)(lstm3)

        # LSTM layer
        lstm_layer1 = LSTM(n_layer, activation='relu')(layer1)
        lstm_layer2 = LSTM(n_layer, activation='relu')(layer2)
        lstm_layer3 = LSTM(n_layer, activation='relu')(layer3)

        # Flattening all the layers
        lstm_output = Concatenate(axis=1)([lstm_layer1, lstm_layer2, lstm_layer3])

        y1 = Dense(n_outputs)(lstm_output)
        y2 = Dense(n_outputs)(lstm_output)
        y3 = Dense(n_outputs)(lstm_output)

        self.model = Model(inputs=[lstm1, lstm2, lstm3], outputs=[y1, y2, y3])
        self.batch = batch 
        self.epochs = epochs
        self.n_layer=n_layer
        self.lr = lr 
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        
    def train(self):
        # Getting the untrained model 
        empty_model = self.model
        
        # Initiating the optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model
        empty_model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizer)

        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch, 
                validation_data=(self.Xval, self.Yval), 
                shuffle=False
            )
        else:
            history = empty_model.fit(
                self.X, 
                self.Y, 
                epochs=self.epochs, 
                batch_size=self.batch,
                shuffle=False
            )
        
        # Saving to original model attribute in the class
        self.model = empty_model
        
        # Returning the training history
        return history
    
    def predict(self, X):
        return self.model.predict(X)