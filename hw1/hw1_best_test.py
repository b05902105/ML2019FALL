from hw1_best_train import *
#from keras.models import load_model
import sys

test_path = sys.argv[1]
output_path = sys.argv[2]

w, b, model, svr_model = None, None, None, None
#model = load_model('nn_model.h5', custom_objects={'rmse': rmse})
with open('best.pkl', 'rb') as f:
    w, b, svr_model = pk.load(f)

hr = 9
fea = 162

test_x = split_test_data(read_data(test_path), fea)
ly = predict(w, b, test_x)
#ny = model.predict(test_x).reshape(-1, 1)
sy = svr_model.predict(test_x).reshape(-1, 1)

cof = [8, 2]
#ey = (cof[0]*ly + cof[1]*ny + cof[2]*sy)/np.sum(cof)
ey = (cof[0]*ly + cof[1]*sy)/np.sum(cof)
output(ey, output_path)
