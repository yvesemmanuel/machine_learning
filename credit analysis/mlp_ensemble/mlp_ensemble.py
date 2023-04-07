import joblib

from keras import Sequential

from keras.layers import Dense
from keras.layers import Dropout

from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import datetime


class MlpEnsemble:

    def __init__(self, models_params, input_dimension):
        self.models_params = models_params
        self.n_members = len(models_params)
        self.input_dimension = input_dimension

        self.base_path = './raw_models/'
        self.sub_models = {}

    def build_sub_model(self, params: dict) -> Sequential:
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

            model.add(Dropout(params['dropout_rate']))

        model.add(
            Dense(
                units=1,
                activation=params['output_activation']
            )
        )

        return model

    def build_sub_models(self):
        build_map = map(self.build_sub_model, self.models_params)
        self.sub_models = list(build_map)

    def train_sub_models(self, X_train, y_train, X_val, y_val):
        histories = list()

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            restore_best_weights=True
        )

        for i, params in enumerate(self.models_params):

            if params['optimizer'] == 'SGD':
                optimizer = SGD(learning_rate=params['alpha'])
            else:
                optimizer = Adam(learning_rate=params['alpha'])

            self.sub_models[i].compile(
                optimizer=optimizer,
                loss=params['loss_function'],
                metrics=['accuracy']
            )

            history = self.sub_models[i].fit(
                X_train,
                y_train,
                batch_size=params['batch_size'],
                epochs=params['max_iter'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping]
            )

            histories.append(history)

        self.histories = histories

    def save_sub_models(self):
        for id, model in enumerate(self.sub_models):
            filename = self.base_path + f'sub_model_{id}.pkl'

            joblib.dump(model, filename)

    def save_model(self):
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = self.base_path + f"ensemble_model_{date_string}.pkl"

        joblib.dump(self, filename)

    def fit(self, X_train, y_train, X_val, y_val):
        self.build_sub_models()
        self.train_sub_models(X_train, y_train, X_val, y_val)
        self.save_sub_models()
        self.fit_stacked_model(X_train, y_train)

    def fit_w_sub_models(self, X_train, y_train):
        all_models = list()

        for i in range(self.n_members):
            filename = self.base_path + f'model_{i}.pkl'

            model = joblib.load(filename)
            all_models.append(model)

        self.sub_models = all_models

        self.fit_stacked_model(X_train, y_train)

    def get_stacked_dataset(self, X):
        X_stacked = self.sub_models[0].predict(X, verbose=0)

        for model in self.sub_models[1:]:
            yhat = model.predict(X, verbose=0)

            X_stacked = np.dstack((X_stacked, yhat))

        X_stacked = X_stacked.reshape((X_stacked.shape[0], X_stacked.shape[1]*X_stacked.shape[2]))

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

    def predict_proba(self, X):
        X_stacked = self.get_stacked_dataset(X)

        return self.model.predict_proba(X_stacked)
