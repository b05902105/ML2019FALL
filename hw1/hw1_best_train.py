from hw1_train import *
import _pickle as pk

from sklearn.svm import SVR

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import tensorflow as tf
from keras import backend as K

def rmse(y_true, y_pred):
    return tf.sqrt(K.mean(K.square(y_pred - y_true)))

if __name__ == '__main__':
    hr = 9
    fea = 162

    tx1, ty1 = split_train_data(read_data('year1-data.csv'), hr)
    tx2, ty2 = split_train_data(read_data('year2-data.csv'), hr)
    train_x, train_y = np.r_[tx1, tx2], np.r_[ty1, ty2]


    model = Sequential()
    model.add(Dense(input_dim=fea, units=500))
    model.add(Dropout(rate = 0.5))
    for i in range(10):
        model.add(Dense(units=500))
        model.add(Dropout(rate = 0.5))
    model.add(Dense(units=1))

    model.compile(loss=rmse, optimizer='adam')
    model.fit(train_x, train_y, batch_size=1000, epochs=300, verbose=2, validation_split=0.1)
    model.evaluate(train_x, train_y)
    y = model.predict(train_x)
    print(loss(y, train_y))

    svr_model = SVR()
    svr_model.fit(train_x, train_y.reshape(-1))

    w, b = best_fit(train_x, train_y)
    print(loss(predict(w, b, train_x),train_y))
    model.save('nn_model.h5')
    
    with open('best.pkl', 'wb') as f:
        pk.dump((w, b, svr_model), f)

    linear_y = predict(w, b, train_x)
    nn_y = model.predict(train_x)
    svr_y = svr_model.predict(train_x)
    cof = [4, 4, 2]
    ensemble_y = (cof[0]*linear_y + cof[1]*nn_y + cof[2]*svr_y)/np.sum(cof)
    print('train error: ', loss(ensemble_y,train_y))