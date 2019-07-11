from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd


def create_model():
    model = Sequential()
    model.add(Dense(126, input_dim=63, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def normalize_data(x):
    sc = StandardScaler()
    return sc.fit_transform(x)


def run():
    data_set = pd.read_csv('data.csv')

    x = data_set.iloc[:, 0:63]
    y = data_set.iloc[:, -1]

    x = normalize_data(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = create_model()
    model.fit(x_train, y_train, batch_size=100, epochs=100)

    eval_model = model.evaluate(x_train, y_train)
    print(eval_model)

    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    model.save('learned_model')
