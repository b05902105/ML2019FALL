import numpy as np
import pandas as pd
import math

def normalize(x):
    return (x - np.mean(x, axis=0))/(np.std(x, axis=0)+1e-100)

def rescaling(x):
    M = np.max(x, axis=0)
    m = np.min(x, axis=0)
    return (x - m)/(M-m+1e-100)

def to_label(y):
    return (y >= 0.5).astype(np.int)

def acc(y, ty):
    return np.average(y == ty)

def output(y, path='./output.csv'):
    idx = np.arange(1, y.shape[0]+1).reshape(-1, 1)
    with open(path, 'w') as f:
        f.write(pd.DataFrame(np.c_[idx, y], columns=['id', 'label']).to_csv(index=False))
