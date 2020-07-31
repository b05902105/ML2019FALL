import pandas as pd
import numpy as np
import _pickle as pk
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import os, sys

def save(path, x):
    with open(path, 'wb') as f:
        pk.dump(x, f) 

def load(path):
    with open(path, 'rb') as f:
        return pk.load(f)
    
def output(y, path='./output.csv'):
    np.savetxt(path, np.c_[np.arange(len(y)), y], delimiter=',', header='id,label', comments='', fmt=['%d', '%d'])
    
def f1_score(pred_y, true_y):
    pre = true_y[pred_y == 1].mean() #classifier預測的正確率
    rec = pred_y[true_y == 1].mean() #真的是惡意留言的情況下，classifier辨別的正確率
    return 2*((pre*rec)/(pre+rec+1e-40))

def plot_Loss_Accuracy_Curves(loss, acc, path='image', name=''):
    epoch = np.arange(len(loss))
    fig = plt.figure(figsize=(6, 6))
    # Loss Curves
    plt.plot(epoch, loss, "r", linewidth=1.5)
    plt.plot(epoch, acc, "b", linewidth=1.5)
    plt.legend(["Training loss", "Training Accuracy"], fontsize=12)
    plt.xlabel("Epochs ", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Loss & Accuracy Curves", fontsize=10)
    plt.savefig(os.path.join(path, name+'LossAccuracyCurves.png'))

def train_word2vec_model(corpus, min_count=1, iter=20, size=256, path='model/word2vec.pkl'):
    model = Word2Vec(min_count=min_count, workers=24, iter=iter, size=size)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    save(path=path, x=model)
    return model
