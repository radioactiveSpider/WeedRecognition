from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


def create_model(input_dim, output_of_hidden_layer):
    model = Sequential()
    model.add(Dense(output_of_hidden_layer, input_dim=input_dim, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_graph_of_acc(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])

    ax.set(xlabel='epoch', ylabel='accuracy',
           title='model accuracy')
    ax.grid()
    ax.legend(['train', 'test'], loc='upper left')

    fig.savefig('model accuracy')


def create_graph_of_loss(history):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])

    ax.set(xlabel='epoch', ylabel='loss',
           title='model loss')
    ax.grid()
    ax.legend(['train', 'test'], loc='upper left')

    fig.savefig('model loss')


def train_and_evaluate_model(input_dim, output_of_hidden_layer, x_train, y_train, x_test, y_test):
    model = create_model(input_dim, output_of_hidden_layer)
    return model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, epochs=100)


def create_columns_names(num_of_epoch):
    names = []
    for i in range(num_of_epoch):
        names.append("epoch" + str(i))
    return names


def create_csv_files(cross_val_arrs):
    for key in cross_val_arrs[0].history.keys():
        filename = str(key) + ".csv"
        with open(filename, "w", newline="") as file:
            names = create_columns_names(len(cross_val_arrs[0].history[key]))
            writer = csv.DictWriter(file, fieldnames=names)
            writer.writeheader()
            for arr in cross_val_arrs:
                writer.writerow(dict(zip(names.copy(), arr.history[key])))


def evaluate_model(data, labels, input_dim, output_of_hidden_layer, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    histories = []
    for train_index, test_index in skf.split(data, labels):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        histories.append(train_and_evaluate_model(input_dim, output_of_hidden_layer, x_train, y_train, x_test, y_test))
    return histories


def create_plots(histories):
    for i, history in enumerate(histories):
        create_graph_of_acc(history, i)
        create_graph_of_loss(history, i)


def normalize_data(x):
    sc = StandardScaler()
    return sc.fit_transform(x)


def find_mean_accuracy(histories):
    val_acc_arr = []
    for history in histories:
        last_val_acc = history.history["val_acc"][len(history.history["val_acc"]) - 1]
        val_acc_arr.append(last_val_acc)
    return np.mean(val_acc_arr)


def get_dataset(data_filename, amount_of_vars):
    data_set = pd.read_csv(data_filename)
    x = data_set.iloc[:, 0:amount_of_vars]
    y = data_set.iloc[:, -1]
    x = normalize_data(x)
    return [x, y]


def get_accuracy_of_model(data_filename, amount_of_bins, hid_layer_neurons):
    x = get_dataset(data_filename, amount_of_bins * 3)[0]
    y = get_dataset(data_filename, amount_of_bins * 3)[1]

    histories = evaluate_model(x, y, amount_of_bins * 3, hid_layer_neurons, 10)
    return find_mean_accuracy(histories)