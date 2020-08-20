#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 10:16:17 2020

class for model. Model includes:
1. FFNN - Feed-Forward Neural Network
2. CNN - Convolutional Neural Network
3. LSTM - Long Short-Term Memory

@author: Farruh Kushnazarov
"""


import pandas as pd
import numpy as np
import pickle
import os

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split




class Modeling:
    cols = 18
    df_result = pd.DataFrame(columns=['FFNN', 'CNN', 'LSTM'],
                             index=['True Negative', 'False Positive', 'False Negative', 'True Positive',
                                    'Accuracy', 'Recall', 'Precision', 'F1_score'])

    def __init__(self, project_path, path_to_data, path_to_results, num_classes=2, epochs=2, batch_size=128):
        self.project_path = project_path
        self.path_to_data = path_to_data
        self.path_to_results = path_to_results
        file_path = path_to_data / 'data.pkl'
        with open(file_path, 'rb') as input_:
            data = pickle.load(input_)
        file_path = path_to_data / 'labels.pkl'
        with open(file_path, 'rb') as input_:
            labels = pickle.load(input_)

        X_train_full, X_test, y_train_full, y_test = train_test_split(data, labels, test_size=0.2)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1)

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_valid = np.array(X_valid)
        self.y_valid = np.array(y_valid)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

    def TrainFFNN(self):
        # Get row, column, and class sizes
        rows = self.X_train.shape[0]
        cols = self.X_train.shape[1]
        val_rows = self.X_valid.shape[0]
        val_cols = self.X_valid.shape[1]
        # input image dimensions to feed into 2D ConvNet Input layer
        input_shape = [cols]
        X_train_FFNN = self.X_train.reshape(rows, cols, 1)
        X_valid_FFNN = self.X_valid.reshape(val_rows, val_cols, 1)

        model = Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dense(300, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dense(600, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dense(100, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dense(self.num_classes, activation="sigmoid")
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-3),
                      metrics=['accuracy'])
        print(model.summary())
        # Stops training if accuracy does not change at least 0.005 over 10 epochs
        checkpoint_cb = ModelCheckpoint((self.project_path / "models/best_FFNN.h5").as_posix(), save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

        # Creates log file for graphical interpretation using TensorBoard
        run_index = 1  # increment this at every run
        run_logdir = os.path.join(os.curdir, "logs/FFNN_logs", "run_{:03d}".format(run_index))
        run_logdir
        tensorboard_cb = TensorBoard(run_logdir, profile_batch=32, write_images=True)

        # Fit model using ImageDataGenerator
        history = model.fit(X_train_FFNN, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=(X_valid_FFNN, self.y_valid),
                            callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])

        return history

    def TrainCNN(self):
        '''
        Trains 2D convolutional neural network
        :param X_train: Numpy array of mfccs
        :param y_train: Binary matrix based on labels
        :return: Trained model
        '''

        # Get row, column, and class sizes
        rows = int(self.X_train.shape[1] / self.cols)
        n_train_samples = self.X_train.shape[0]
        n_valid_samples = self.X_valid.shape[0]
        # input image dimensions to feed into 2D ConvNet Input layer
        input_shape = (rows, self.cols, 1)
        X_train_CNN = self.X_train.reshape(n_train_samples, rows, self.cols, 1)
        X_valid_CNN = self.X_valid.reshape(n_valid_samples, rows, self.cols, 1)

        model = Sequential([
            Conv2D(64, 7, activation="relu", padding="same",
                   input_shape=input_shape),
            MaxPooling2D(2),
            Conv2D(128, 3, activation="relu", padding="same"),
            Conv2D(128, 3, activation="relu", padding="same"),
            MaxPooling2D(2),
            Conv2D(256, 3, activation="relu", padding="same"),
            Conv2D(256, 3, activation="relu", padding="same"),
            MaxPooling2D(2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(units=self.num_classes, activation='softmax'),
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
        print(model.summary())
        # Stops training if accuracy does not change at least 0.005 over 10 epochs
        checkpoint_cb = ModelCheckpoint((self.project_path / "models/best_CNN.h5").as_posix(), save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

        # Creates log file for graphical interpretation using TensorBoard
        run_index = 1  # increment this at every run
        run_logdir = os.path.join(os.curdir, "logs/CNN_logs", "run_{:03d}".format(run_index))
        run_logdir
        tensorboard_cb = TensorBoard(run_logdir, profile_batch=32, write_images=True)

        # Fit model using ImageDataGenerator
        history = model.fit(X_train_CNN, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            validation_data=(X_valid_CNN, self.y_valid),
                            callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])

        return history

    def TrainLSTM(self):
        rows = int(self.X_train.shape[1] / self.cols)
        n_train_samples = self.X_train.shape[0]
        n_valid_samples = self.X_valid.shape[0]
        # input image dimensions to feed into 2D ConvNet Input layer
        input_shape = (rows, self.cols, 1)
        X_train_LSTM = self.X_train.reshape(n_train_samples, rows, self.cols)
        X_valid_LSTM = self.X_valid.reshape(n_valid_samples, rows, self.cols)
        lstm = Sequential([
            LSTM(64, return_sequences=True, stateful=False, activation='tanh'),
            LSTM(64, stateful=False, activation='tanh'),
            # add dropout to control for overfitting
            Dropout(.25),
            # squash output onto number of classes in probability space
            Dense(self.num_classes, activation='softmax'),
        ])
        adam = optimizers.Adam(lr=0.0001)
        lstm.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

        # Stops training if accuracy does not change at least 0.005 over 10 epochs
        checkpoint_cb = ModelCheckpoint((self.project_path / "models/best_LSTM.h5").as_posix(), save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

        # Creates log file for graphical interpretation using TensorBoard
        run_index = 1  # increment this at every run
        run_logdir = os.path.join(os.curdir, "logs/LSTM_logs", "run_{:03d}".format(run_index))
        run_logdir

        # Fit model using ImageDataGenerator
        history = lstm.fit(X_train_LSTM, self.y_train,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_data=(X_valid_LSTM, self.y_valid),
                           callbacks=[early_stopping_cb, checkpoint_cb]
                           )
        return history

    def PredictionResults(self, y_prediction):
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_prediction.round()).ravel()
        result = [tn, fp, fn, tp,
                  round(accuracy_score(self.y_test, y_prediction.round()), 2),
                  round(recall_score(self.y_test, y_prediction.round()), 2),
                  round(precision_score(self.y_test, y_prediction.round()), 2),
                  round(precision_score(self.y_test, y_prediction.round()), 2)
                  ]
        return result

    def MultiModel(self, network, need_to_train=False):
        if need_to_train:
            methods = {'trainModelFFNN': self.TrainFFNN,
                       'trainModelCNN': self.TrainCNN,
                       'trainModelLSTM': self.TrainLSTM}
            method_name = 'trainModel' + network  # set by the command line options
            if method_name in methods:
                history = methods[method_name]()  # + argument list of course
            else:
                raise Exception("Method %s not implemented" % method_name)

        model = load_model(self.project_path / ("models/best_" + network + ".h5"))  # rollback to best mod

        plot_model(
            model,
            to_file="../img/model_" + network + "_architecture.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=900,
        )

        if network == 'FFNN':
            rows = self.X_test.shape[0]
            cols = self.X_test.shape[1]
            X_test_FFNN = self.X_test.reshape(rows, cols, 1)
            y_pred = model.predict_classes(X_test_FFNN)
        elif network == 'CNN':
            cols = self.cols
            rows = int(self.X_test.shape[1] / cols)
            n_test_samples = self.X_test.shape[0]
            X_test_CNN = self.X_test.reshape(n_test_samples, rows, cols, 1)
            y_pred = model.predict_classes(X_test_CNN)
        elif network == 'LSTM':
            cols = self.cols
            rows = int(self.X_test.shape[1] / cols)
            n_test_samples = self.X_test.shape[0]
            X_test_LSTM = self.X_test.reshape(n_test_samples, rows, cols)
            y_pred = model.predict_classes(X_test_LSTM)

        self.df_result[network] = self.PredictionResults(y_pred)

        df_prediction = pd.DataFrame({
            'Actual value': self.y_test,
            'Predicted value': y_pred
        })

        file_path = self.path_to_results / ('df_prediction_' + network + '.csv')
        df_prediction.to_csv(file_path, sep='\t', encoding='utf-8')

    def PrintResult(self):
        print(self.df_result)
    def SaveResults(self):
        file_path = self.path_to_results / "df_results.csv"
        self.df_result.to_csv(file_path, sep='\t', encoding='utf-8')

