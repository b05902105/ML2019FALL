import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os, sys

import spacy
from process_bar import ShowProcess
from util import *
from model import *
from itertools import groupby

def clean_data(seq):
    return [x[0] for x in groupby(seq)]

def seq_padding(x, l, mode='pre'):
    n = x.shape[0]
    z = np.zeros((l, x[0].shape[0]))
    if mode == 'post':
        z[:n] = x
    elif mode == 'pre':
        z[-n:] = x
    elif mode == 'avg':
        z[l//2 - n//2: l//2 - n//2 + n] = x
    return z

class Data(Dataset):
    def __init__(self, corpus, word_model, padding_length=80):
        self.padding_length = padding_length
        self.corpus = corpus
        self.word2vec = word_model
    def __len__(self):
        return len(self.corpus)
    def __getitem__(self, idx):
        seq = seq_padding(np.array([self.word2vec.wv[w] for w in self.corpus[idx]]), self.padding_length)
        return seq
    
def predict(test_loader, model):
    model.eval()
    pred_y = []
    with torch.no_grad():
        process_bar = ShowProcess(len(test_loader))
        for i, d in enumerate(test_loader):
            seq = torch.transpose(d, 1, 0).float().cuda()
            output = model(seq)
            pred_y.append(torch.argmax(output.data, dim=1).cpu().numpy().astype(np.int))
            process_bar.show_process()
    return np.concatenate(pred_y, axis=0)

def voting(*args):
    result = np.zeros(args[0].shape)
    for a in args:
        result += (a*2 - 1)
    return ((np.sign(result) + 1)/2).astype(np.int) 

if __name__ == '__main__':
    hidden_dim = 128
    path = sys.argv[1]
    output_path = sys.argv[2]
    
    test_x = pd.read_csv(path, header=0).to_numpy()[:, 1]
    
    test_corpus = load('model/test_corpus.pkl')
    
    #test_corpus = []
    #nlp = spacy.load("en_core_web_sm")
    #for s in test_x:
    #    doc = nlp(s)
    #    token = clean_data([t.text.lower() for t in doc])
    #    test_corpus.append(token)
    
    word_model = load('model/word2vec.pkl')
    test_d = Data(test_corpus, word_model, padding_length=86)
    test_loader = DataLoader(test_d, batch_size=64, shuffle=False)
    
    model1 = LSTMClassifier(embedding_dim=256, hidden_dim=hidden_dim).cuda()
    model2 = model_2(embedding_dim=256, hidden_dim=hidden_dim).cuda()
    model3 = model_3(embedding_dim=256, hidden_dim=256).cuda()
    model1.load_state_dict(torch.load('model/model.mdl'))
    model2.load_state_dict(torch.load('model/model2.mdl'))
    model3.load_state_dict(torch.load('model/model3.mdl'))
    
    pred_y1 = predict(test_loader, model1)
    pred_y2 = predict(test_loader, model2)
    pred_y3 = predict(test_loader, model3)
    
    output(voting(pred_y1, pred_y2, pred_y3), output_path)
