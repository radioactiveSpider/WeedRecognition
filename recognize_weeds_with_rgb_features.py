import numpy as np
import handle_data
import MLP_binary_classification
from keras.models import load_model
import evaluate_models


def get_prediction_for_one_pic(x, model):
    y_pred = model.predict(x)
    y_pred = [y_pred > 0.5]
    return np.sum(y_pred) / np.asarray(y_pred).shape[1]


def run():
    model = load_model('learned_model.h5')
    params = evaluate_models.find_best_model("config.csv")

    handle_data.run('data.csv', 'dataset-master/images', 'dataset-master/annotations',
                    int(params['no_of_tiles_x']), int(params['no_of_tiles_y']),
                    int(params['no_of_bins']), int(params['median_filter_param']))

    x, y = MLP_binary_classification.get_dataset('data.csv', int(params['no_of_bins']) * 3)
    amount_of_tiles_in_pic = int(params['no_of_tiles_x']) * int(params['no_of_tiles_y'])

    shift_1 = 0
    shift_2 = amount_of_tiles_in_pic
    for i in range(int(x.shape[0] / amount_of_tiles_in_pic)):
        print("No of pic is\n")
        print(i)

        one_pic = x[shift_1:shift_2, :]
        prediction = get_prediction_for_one_pic(one_pic, model)
        #print("Level of weeds is\n")
        #print(prediction)
        print("Is there high level of weeds?\n")
        if prediction == 0:
            print("No weeds\n")
        elif prediction > 0.3:
            print("Yes\n")
        else:
            print("No\n")

            shift_1 += amount_of_tiles_in_pic
            shift_2 += amount_of_tiles_in_pic