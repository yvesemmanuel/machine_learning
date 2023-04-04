import joblib

from keras import Sequential

from keras.layers import Dense
from keras.layers import Dropout

from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np


class MlpEnsemble:

    def __init__(self, models_params, input_dimension):
        self.models_params = models_params
        self.n_members = len(models_params)
        self.input_dimension = input_dimension

        self.base_path = './raw_models/'
        self.models = {}

    def build_models(self):
        for i, params in enumerate(self.models_params):
            model = Sequential()

            model.add(
                Dense(
                    input_dim=self.input_dimension,
                    units=params['hidden_layer_units'],
                    activation=params['activation']
                )
            )

            for _ in range(params['hidden_layers']):
                model.add(
                    Dense(
                        units=params['hidden_layer_units'],
                        activation=params['activation']
                    )
                )

                # model.add(Dropout(params['dropout_rate']))


            model.add(
                Dense(
                    units=1,
                    activation=params['output_activation']
                )
            )

            self.models[i] = model

    def train_models(self, X_train, y_train, X_val, y_val):    
        histories = list()

        for i, params in enumerate(self.models_params):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=1,
                restore_best_weights=True
            )

            if params['optimizer'] == 'SGD':
                optimizer = SGD(learning_rate=params['learning_rate'])
            elif params['optimizer'] == 'Adam':
                optimizer = Adam(learning_rate=params['learning_rate'])

            self.models[i].compile(
                optimizer=optimizer,
                loss=params['loss_function'],
                metrics=['accuracy']
            )

            history = self.models[i].fit(
                X_train,
                y_train,
                batch_size=params['batch_size'],
                epochs=params['max_iter'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping]
            )

            histories.append(history)

        self.histories = histories

    def save_models(self):
        for id in self.models:
            filename = self.base_path + f'model_{id}.pkl'

            joblib.dump(self.models[id], filename)

    def fit(self, X_train, y_train, X_val, y_val):
        self.build_models()
        self.train_models(X_train, y_train, X_val, y_val)
        self.save_models()
        # self.set_models()
        
        # print(f'{self.n_members} created at {self.base_path}.')

        # self.fit_stacked_model(X_train, y_train)

    def set_models(self):
        all_models = list()

        for i in range(self.n_members):
            filename = self.base_path + f'model_{i}.pkl'

            model = joblib.load(filename)
            all_models.append(model)

        self.models = all_models

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
