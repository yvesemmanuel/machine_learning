import joblib

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np


class MLP_Ensemble:

    def __init__(self, n_members: int = 1, hidden_layer_sizes: int = 1, learning_rate: float = 0.01, max_iter: int = 10E3, batch_size: int = 8, activation: str = 'relu'):
        self.n_members = n_members
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.activation = activation

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
        for i in range(self.n_members):
            model = keras.Sequential([
                keras.layers.Dense(
                    units=self.hidden_layer_sizes,
                    activation=self.activation
                ),
                keras.layers.Dense(
                    units=self.hidden_layer_sizes,
                    activation=self.activation
                ),
                keras.layers.Dense(units=1, activation='softmax')
            ])

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            )

            model.compile(
                optimizer=keras.optimizers.Adam(
                    self.learning_rate
                ),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=['accuracy']
            )

            model.fit(
                X,
                y,
                batch_size=self.batch_size,
                epochs=self.max_iter,
                callbacks=[early_stopping],
                validation_data=(X_val, y_val)
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
