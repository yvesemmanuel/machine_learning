import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib

import numpy as np


class EfficientNet():

    def __init__(self, img_dimension=224, seed=0, activation_function='relu', output_function='sigmoid') -> None:

        self.base_path = './raw_models/'

        pretrained_efficientnet_base = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False,
            input_shape=(img_dimension, img_dimension, 3),
            weights='imagenet',
            pooling='avg',
        )
        pretrained_efficientnet_base.trainable = False

        self.model = tf.keras.Sequential([
            layers.Input(shape=(img_dimension, img_dimension, 3)),
            
            layers.RandomBrightness(0.2, seed=seed),
            layers.RandomFlip(seed=seed),
            layers.RandomRotation(0.2, seed=seed),
            
            layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
            pretrained_efficientnet_base,
            layers.Dropout(0.5),
            
            layers.Dense(256, activation=activation_function),
            layers.Dropout(0.2),
            layers.Dense(32, activation=activation_function),
            layers.Dense(1, activation=output_function)
        ], name='EfficientNet')


    def fit(
        self,
        training_data,
        validation_data,
        learning_rate=0.01,
        patience=5,
        epochs=100,
        verbose=0
    ) -> tf.keras.callbacks.History:
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name='roc_auc'), 'binary_accuracy']
        )

        early_stopping = EarlyStopping(
            min_delta=1e-4,
            patience=patience,
            verbose=verbose,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=patience, verbose=verbose)

        self.history = self.model.fit(
            training_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr]
        )
        
    
    def show_history(self):
        performance_df = pd.DataFrame(self.history)
        fig, axes = plt.subplots(ncols=2, figsize=(11, 4))

        for ax, metric in zip(axes.flat, ['Accuracy', 'Loss']):
            performance_df.filter(like=metric.lower()).plot(ax=ax)
            ax.set_title(metric, size=14, pad=10)
            ax.set_xlabel('epoch')


    def predict_proba(self, X):
        X_preprocessed = tf.keras.applications.vgg16.preprocess_input(X)

        return self.model.predict(X_preprocessed)


    def predict(self, test_data):
        results = [(labels, self.model.predict(images, verbose=0).reshape(-1)) for images, labels in test_data.take(-1)]

        labels = np.concatenate([x[0] for x in results])
        preds = np.concatenate([x[1] for x in results])

        return labels, preds

    
    def evaluate(self, test_data, verbose=0):
        
        return self.model.evaluate(test_data, verbose=verbose)


    def save_model(self, model_name: str):
        filename = self.base_path + f'{model_name}.pkl'

        joblib.dump(self, filename)
