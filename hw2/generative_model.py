from util import *
import sys

def max_like(X):
    m = np.average(X, axis=0)
    s = (X-m).T.dot(X-m) / X.shape[0]
    return m, s

def fit(X, Y):
    Y = Y.reshape(-1)
    c1, c2 = np.average(Y==0), np.average(Y==1)
    m1, s1 = max_like(X[Y==0])
    m2, s2 = max_like(X[Y==1])
    share_s = c1 * s1 + c2 * s2
    return (c1, m1), (c2, m2), share_s

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(p, X):
    (c1, m1), (c2, m2), s = p
    inv = np.linalg.inv(s)
    w = (m1-m2).dot(inv).reshape(-1, 1)
    b = -0.5 * (m1.T.dot(inv).dot(m1)) + 0.5 * m2.T.dot(inv).dot(m2) + np.log(c1/c2)
    z = X.dot(w) + b
    return (sigmoid(z) < 0.5).astype(np.int)

if __name__ == '__main__':
    is_train = False  
    x_train_path = sys.argv[3]
    y_train_path = sys.argv[4]
    test_path = sys.argv[5]
    output_path = sys.argv[6]

    x_train = pd.read_csv(x_train_path, header=0).to_numpy()
    y_train = pd.read_csv(y_train_path, header=None).to_numpy()
    x_test = pd.read_csv(test_path, header=0).to_numpy()
    
    scal_x_train = rescaling(x_train)
    scal_x_test = rescaling(x_test)
    
    if is_train:
        p1, p2, share_s = np.load('./model/generative_model.npy', allow_pickle=True)
    else:
        p1, p2, share_s = fit(scal_x_train, y_train.reshape(-1))
        np.save('./model/generative_model.npy', (p1, p2, share_s), allow_pickle=True)

    pred_y = predict((p1, p2, share_s), scal_x_train)
    print('training accuracy: %.4f' % acc(pred_y, y_train))
    out_y = predict((p1, p2, share_s), scal_x_test)
    output(out_y, output_path)
