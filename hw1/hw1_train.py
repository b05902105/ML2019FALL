import numpy as np
import pandas as pd
import math

def read_data(name):
    df = pd.read_csv(name).iloc[:, 2:]
    df.fillna(0)
    d = df.applymap(lambda x: str(x).rstrip('*#xA')).to_numpy()
    d[d == 'NR'] = 0
    d[d == ''] = 0
    d[d == 'nan'] = 0
    return d.astype(np.float)

def split_train_data(data, r):
    X, Y = [], []
    for i in range(0, len(data), 18):
        d = data[i:i+18].T
        for j in range(len(d)-r):
            #取10個小時的資料
            #X為取前9列作其他操作, Y是第10列的pm2.5
            tmp_d = d[j:j+r+1]
            if (tmp_d[:, :9] <= 0).any() or (tmp_d[:, 11:] <= 0).any()\
            or(tmp_d[:, 9] < 2).any() or (tmp_d[:, 9] > 100).any()\
            or (tmp_d[:, 8] < 2).any() or (tmp_d[:, 8] > 100).any():
                continue
            Y.append(tmp_d[-1, 9])
            X.append(tmp_d[:-1].reshape(-1))
    return np.array(X), np.array(Y).reshape(-1, 1)

def split_test_data(data, fea):
    X = np.zeros(fea).reshape(1, -1)
    for i in range(0, len(data), 18):
        d = data[i:i+18].T
        
        x = d.reshape(1, -1)
        X = np.r_[X, x]
    return X[1:]

def loss(y, true_y):
    return np.sqrt(np.average((y-true_y)**2))

def predict(w, b, x):
    return x.dot(w.reshape(-1, 1))+b

def fit(x, y, w=None, b=0, iter=10, lr=1, lamb=0, p=True):
    N = len(x)
    h = len(x[0])
    if type(w) != type(np.array([])):
        w = np.array([0.1]*h).reshape(-1, 1)
    lr_w = 0
    lr_b = 0
    for i in range(iter):
        w_grad = np.average(2*(x.dot(w)+b - y)*x, axis=0).reshape(-1, 1) + lamb*w
        b_grad = np.average(2*(x.dot(w)+b - y))
        lr_w += w_grad**2
        lr_b += b_grad**2
        w = w - lr/np.sqrt(lr_w) * w_grad
        b = b - lr/np.sqrt(lr_b) * b_grad
    if p:
        print('loss:', loss(predict(w, b, x), y))
    return w, b

def minibatch(x, y):
    # 打亂data順序
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    # 訓練參數以及初始化
    batch_size = 64
    lr = 1e-3
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(x[0].shape, 0.1).reshape(-1, 1)
    bias = 0.1
    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    
    for num in range(1000):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            # 計算gradient
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
            # 更新weight, bias
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
            

    return w, bias

def k_fold_cross_validtion(X, Y, k=10):
    N = len(X)
    dim = len(X[0])
    index = np.arange(N)
    np.random.shuffle(index)
    val_err = 0
    for i in range(k):
        s = i*(N//k) + i*(i < N%k) + (N%k)*(i >= N%k)
        d = s + N//k + (i < N%k)
        val_x, val_y = X[index[s:d]], Y[index[s:d]]
        x = np.r_[X[index[:s]], X[index[d:]]]
        y = np.r_[Y[index[:s]], Y[index[d:]]]
        w, b = best_fit(x, y)
        
        cof = [8, 2]
        ly = predict(w, b, val_x)
        l = loss(ly, val_y)
        print('loss:', l)
        val_err += l
    return val_err / k

def bootstrap(X, Y, test_x, k=10, rate=0.7):
    N = len(X)
    val_err = 0
    uniform_bagging_err = 0
    train_bag_y = np.zeros(len(Y)).reshape(-1, 1)
    ret_y = np.zeros(len(test_x)).reshape(-1, 1)
    for i in range(k):
        index = np.arange(N)
        np.random.shuffle(index)
        val_idx = int(N*rate)
        val_x, val_y = X[index[val_idx:]], Y[index[val_idx:]]
        x, y = X[index[:val_idx]], Y[index[:val_idx]]
        w, b = best_fit(x, y)
        val_err = loss(predict(w, b, val_x), val_y)
        print('model {0} validation error: '.format(i), val_err)
        
        uy = predict(w, b, X)
        train_bag_y += uy
        uniform_bagging_err += loss(uy, Y)
        ret_y += predict(w, b, test_x)
    print('training error: ', uniform_bagging_err/k)
    
    return train_bag_y/k, ret_y/k

def z_transform(x):
    h = x.shape[1]
    z = np.zeros(2*h).reshape(1, -1)
    for i in x:
        t = np.r_[np.diag(i), np.identity(h)].dot(i).reshape(1, -1)
        z = np.r_[z, t]
    return z[1:]

def best_fit(x, y):
    o = np.ones(len(x)).reshape(-1, 1)
    x = np.c_[o, x]
    sol = np.linalg.pinv(x).dot(y)
    return sol[1:].reshape(-1), sol[0]

def output(y, path='./output.csv'):
    table = [['id_'+str(i), a] for i, a in enumerate(y.flatten())]
    with open(path, 'w') as f:
        f.write(pd.DataFrame(table, columns = ['id', 'value']).to_csv(index=False))

def save(w, b, name):
    with open(name + '.pkl', 'wb') as f:
        pk.dump((w, b), f)

if __name__ == '__main__':
    hr = 9
    fea = 162

    tx1, ty1 = split_train_data(read_data('year1-data.csv'), hr)
    tx2, ty2 = split_train_data(read_data('year2-data.csv'), hr)
    train_x, train_y = np.r_[tx1, tx2], np.r_[ty1, ty2]
    
    #k_fold_cross_validtion(train_x, train_y)

    w, b = best_fit(train_x, train_y)
    print(loss(predict(w, b, train_x),train_y))

    #train_bag_y, test_bag_y = bootstrap(train_x, train_y, test_x)
    
    np.savez("./hw1.npz", w, np.array(b))