import joblib

from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np


class MLP_Ensemble:

    def __init__(self, n_members: int = 1, hidden_layer_sizes: int = 1, hidden_layers: int = 1, learning_rate: float = 0.01, max_iter: int = 10E3, batch_size: int = 8, activation: str = 'relu', output_activation: str = 'softmax', optimizer: str = 'SGD', loss_function: str = 'binary_crossentropy'):
        self.n_members = n_members
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.hidden_layers = hidden_layers
        self.loss_function = loss_function

        self.base_path = '../models/'
        self.models = []

    def set_models(self):
        all_models = list()

        for i in range(self.n_members):
            filename = self.base_path + str(i + 1) + '.pkl'

            model = joblib.load(filename)
            all_models.append(model)

        self.models = all_models

    def fit(self, X, y, X_val, y_val):
        input_dimension = X.shape[1]

        for i in range(self.n_members):

            model = Sequential()

            model.add(
                Dense(
                    input_dim=input_dimension,
                    units=self.hidden_layer_sizes,
                    activation=self.activation
                )
            )

            for _ in range(self.hidden_layers):
                model.add(
                    Dense(
                        units=self.hidden_layer_sizes,
                        activation=self.activation
                    )
                )


            model.add(
                Dense(
                    units=1,
                    activation=self.output_activation
                )
            )

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=1,
                restore_best_weights=True,
                min_delta=0.001
            )

            if self.optimizer == 'SGD':
                optimizer = SGD(learning_rate=self.learning_rate)
            elif self.optimizer == 'Adam':
                optimizer = Adam(learning_rate=self.learning_rate)

            model.compile(
                optimizer=optimizer,
                loss=self.loss_function,
                metrics=['accuracy']
            )

            model.fit(
                X,
                y,
                batch_size=self.batch_size,
                epochs=self.max_iter,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping]
            )

            filename = self.base_path + str(i + 1) + '.pkl'

            joblib.dump(model, filename)

        self.set_models()
        print(f'{self.n_members} created at {self.base_path}.')

        self.fit_stacked_model(X,y)

    def predict(self, X, y):
        y_pred = [model.predict(X, verbose=0) for model in self.models]

        acc = accuracy_score(y, y_pred)

        return acc

    def predict1(self, X_test):
        y_preds = np.array([model.predict(X_test) for model in self.models])

        y_ensemble = np.mean(y_preds, axis=0)

        y_pred = np.argmax(y_ensemble, axis=1)

        return y_pred

    def get_stacked_dataset(self, X):
        X_stacked = None

        for model in self.models:
            yhat = model.predict(X, verbose=0)

            if X_stacked is None:
                X_stacked = yhat
            else:
                X_stacked = np.dstack((X_stacked, yhat))

        n, m = X_stacked.shape

        X_stacked = X_stacked.reshape((n, n*m))

        return X_stacked

    def fit_stacked_model(self, X, y):
        X_stacked = self.get_stacked_dataset(X)

        meta_learner = LogisticRegression()
        meta_learner.fit(X_stacked, y)

        self.model = meta_learner

    def stacked_prediction(self, X):
        X_stacked = self.get_stacked_dataset(X)

        yhat = self.model.predict(X_stacked)

        return yhat

    def get_score(self, X, y):
        yhat = self.stacked_prediction(X)

        return accuracy_score(y, yhat)
