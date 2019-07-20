import MLP_binary_classification
from keras.models import load_model
import handle_data
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import csv


def create_config(filename):
    amount = 5
    tiles_x = [3, 3, 3, 4, 3]
    tiles_y = [3, 4, 5, 4, 7]
    bins = [21, 42, 84, 168, 256]
    median_filter_params = [3, 5, 7, 9, 11]

    columns = ["no_of_tiles_x", "no_of_tiles_y", "no_of_bins", "median_filter_param", "no_of_hid_layer_neurons",
               "accuracy"]
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for i in range(amount):
            for j in range(amount):
                for z in range(amount):
                    for h in range(amount):
                        dict = {}
                        dict["no_of_tiles_x"] = tiles_x[i]
                        dict["no_of_tiles_y"] = tiles_y[i]
                        dict["no_of_bins"] = bins[j]
                        dict["median_filter_param"] = median_filter_params[z]
                        dict["no_of_hid_layer_neurons"] = bins[j] * 3 * 2 * (h + 1)
                        dict["accuracy"] = 0.0
                        writer.writerow(dict)
    return filename


def fill_accuracies(filename, type_of_features, data_filename, no_of_epoch, batch_len):
    params = pd.read_csv(filename)
    for i in range(320, 340):
        data_filename = handle_data.run(type_of_features, data_filename, 'dataset-master/images', 'dataset-master/annotations',
                                        int(params.iloc[i]['no_of_tiles_x']), int(params.iloc[i]['no_of_tiles_y']),
                                        int(params.iloc[i]['no_of_bins']), int(params.iloc[i]['median_filter_param']))
        accuracy = MLP_binary_classification.get_accuracy_of_model(data_filename, int(params.iloc[i]["no_of_bins"]),
                                                                   int(params.iloc[i]["no_of_hid_layer_neurons"]),
                                                                   no_of_epoch, batch_len)
        print("iter_" + str(i))
        print("accuracy is " + str(accuracy))
        params.at[i, "accuracy"] = accuracy
    params.to_csv(filename, index=False)


def find_best_model(filename):
    params = pd.read_csv(filename)
    return params[params["accuracy"] == params["accuracy"].max()]


def learn_and_save_best_model(params, data_filename, no_of_epoch, batch_len):
    data_filename = handle_data.run("rgb", data_filename, 'dataset-master/images', 'dataset-master/annotations',
                                    int(params['no_of_tiles_x']), int(params['no_of_tiles_y']),
                                    int(params['no_of_bins']), int(params['median_filter_param']))

    x, y = MLP_binary_classification.get_dataset(data_filename, int(params['no_of_bins']) * 3)

    model = MLP_binary_classification.create_model(int(params['no_of_bins']) * 3, int(params['no_of_hid_layer_neurons']))
    model.fit(x, y, batch_size=batch_len, epochs=no_of_epoch)
    model.save('learned_model.h5')


def get_prediction_for_test_set(model_filename, x, y):
    model = load_model(model_filename)

    y_pred = model.predict(x)
    y_pred = (y_pred > 0.5)

    cm = confusion_matrix(y, y_pred)
    print("confusion matrix\n")
    print(cm)
    print("accuracy of prediction is:\n")
    print((cm[0][0] + cm[1][1]) / np.sum(cm))


def create_graph_for_best_model(model_name, params, x, y, no_of_epoch, batch_len):
    if model_name == "aco":
        model = MLP_binary_classification.create_ann_aco_model(4)
    elif model_name == "rgb":
        model = MLP_binary_classification.create_rgb_model(int(params['no_of_bins']) * 3, int(params['no_of_hid_layer_neurons']))
    else:
        return -1
    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=no_of_epoch, batch_size=batch_len)
    MLP_binary_classification.create_graph_of_acc(history)
    MLP_binary_classification.create_graph_of_loss(history)


def run_rgb_features_models(config_filename, data_filename, test_data_filename):
    #fill_accuracies("rgb", config_filename, "data.csv", 100, 100)
    params = find_best_model(config_filename)
    print("best params is:\n")
    print(params)
    learn_and_save_best_model(params, data_filename, 100, 100)

    handle_data.run("rgb", test_data_filename, 'dataset-master/test_images', 'dataset-master/test_annotations',
                    int(params['no_of_tiles_x']), int(params['no_of_tiles_y']),
                    int(params['no_of_bins']), int(params['median_filter_param']))
    x, y = MLP_binary_classification.get_dataset('test_data.csv', int(params['no_of_bins']) * 3)

    create_graph_for_best_model(params, x, y, 100, 100)
    get_prediction_for_test_set('learned_model.h5', x, y)


def run_glcm_yiq_features_models(data_filename, test_filename, batch_len, no_of_epoch):
    accuracy = MLP_binary_classification.get_accuracy_of_model("aco", data_filename, 4, 0, 100, 100)
    print("accuracy of ann-aco is:\n")
    print(accuracy)

    x, y = MLP_binary_classification.get_dataset(data_filename, 4)
    model = MLP_binary_classification.create_ann_aco_model(4)
    model.fit(x, y, batch_size=batch_len, epochs=no_of_epoch)
    model.save('learned_aco_model.h5')

    test_data_filename = handle_data.run("glcm", test_filename, 'dataset-master/test_images',
                                         'dataset-master/test_annotations', 3, 5, 0, 0)
    x, y = MLP_binary_classification.get_dataset(test_data_filename, 4)
    create_graph_for_best_model("aco", 0, x, y, 100, 100)
    get_prediction_for_test_set('learned_aco_model.h5', x, y)


run_glcm_yiq_features_models("glcm_yiq_data.csv", "test_aco_ann.csv", 100, 100)