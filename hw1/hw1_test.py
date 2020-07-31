from hw1_train import *

import sys

test_path = sys.argv[1]
output_path = sys.argv[2]

npzfile = np.load('./hw1.npz')

hr = 9
fea = 162

w = npzfile['arr_0']
b = npzfile['arr_1'][0]

test_x = split_test_data(read_data(test_path), fea)
test_y = predict(w, b, test_x)
output(test_y, output_path)