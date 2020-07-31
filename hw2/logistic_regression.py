from util import *
import sys

def extract(x, idx):
    return x[:, idx]

def z_transform(x, p=10):
    org_x = x[:, :]
    tmp = x[:, :]
    for i in range(p-1):
        tmp *= org_x
        x = np.c_[x, tmp]
    return x

def fit(x, y, w=None, b=0, iter=1000, lr=1, lamb = 0.01, verbose=1):
    N, h = x.shape
    if type(w) != type(np.array([])):
        w = np.array([1]*h).reshape(-1, 1)
    lr_w = 0
    for i in range(iter):
        grad = x.T.dot(predict(w, x) - y)/N + lamb * w
        lr_w += grad**2
        w = w - lr/np.sqrt(lr_w) * grad
        if verbose == 2:
            py = predict(w, x)
            print('iter: {2}, loss: {0}, acc: {1:.2f}'.format(loss(py, y), acc(to_label(py), y), i))
    if verbose >= 1:
        print('training error: {0}'.format(loss(predict(w, x), y)))
        print('acc: {0:.2f}'.format(acc(to_label(predict(w, x)), y)))
    return w

def predict(w, x):
    z = x.dot(w)
    return 1/(1 + np.exp(-z))

def loss(y, ty):
    ret = -(ty.T.dot(np.log(y+1e-20)) + (1-ty).T.dot(np.log(1-y + 1e-20)))/len(y)
    return ret[0, 0]

if __name__ == '__main__':
    is_train = True
    
    x_train_path = sys.argv[3]
    y_train_path = sys.argv[4]
    test_path = sys.argv[5]
    output_path = sys.argv[6]

    x_train = pd.read_csv(x_train_path, header=0).to_numpy()
    y_train = pd.read_csv(y_train_path, header=None).to_numpy()
    x_test = pd.read_csv(test_path, header=0).to_numpy()
    
    scal_x_train = rescaling(x_train)
    scal_x_test = rescaling(x_test)

    ext_x_train = extract(scal_x_train, [0, 1, 5, 9, 63, 102])
    ext_x_test = extract(scal_x_test, [0, 1, 5, 9, 63, 102])

    z_train = np.c_[z_transform(ext_x_train, 50)[:, 6:], scal_x_train]
    z_test = np.c_[z_transform(ext_x_test, 50)[:, 6:], scal_x_test]

    if is_train:
        w = np.load('./model/log_model.npy', allow_pickle=True)
    else:
        w = fit(z_train, y_train, lamb=0, iter=5000, verbose=2)
        np.save('./model/log_model.npy', w, allow_pickle=True)
    
    pred_y = to_label(predict(w, z_train))
    print('training accuracy: %.4f' % acc(pred_y, y_train))
    output(to_label(predict(w, z_test)), output_path)