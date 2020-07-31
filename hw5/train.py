import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os, sys

import spacy
from process_bar import ShowProcess
from util import *
from model import *

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
    def __init__(self, path, label_path, word_model, padding_length=80):
        self.padding_length = padding_length
        self.corpus = np.load(path, allow_pickle=True)
        self.label = pd.read_csv(label_path, header=0).to_numpy()[:, 1]
        self.word2vec = word_model
    def __len__(self):
        return len(self.corpus)
    def __getitem__(self, idx):
        seq = seq_padding(np.array([self.word2vec.wv[w] for w in self.corpus[idx]]), self.padding_length)
        label = self.label[idx]
        return seq, label

def train(train_loader, model, optimizer, criterion, num_epochs):
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        process_bar = ShowProcess(len(train_loader))
        model.train()
        f1_acc_cnt = 0
        loss_cnt = 0
        for i, d in enumerate(train_loader):
            seq, label = d
            seq = torch.transpose(seq, 1, 0).float().cuda()
            label = label.cuda()
            output = model(seq)
            loss = criterion(output, label)
            loss_cnt += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output.data, dim=1)
            
            f1_acc_cnt += f1_score(pred.cpu().numpy(), label.cpu().numpy())
            print('Epoch [%03d/%03d], ' % (epoch+1, num_epochs), end='')
            process_bar.show_process(other=', acc: %.2f' % (f1_acc_cnt/(i+1)))
        loss_list.append(loss_cnt/(len(train_loader)))
        acc_list.append(f1_acc_cnt/(len(train_loader)))
    return loss_list, acc_list

if __name__ == '__main__':
    path = sys.argv[1]
    label_path = sys.argv[2]

    batch_size = 64
    hidden_dim = 128
    num_epochs = 50
    is_save = False
    
    train_x = pd.read_csv(path, header=0).to_numpy()[:, 1]
    word_model = load('model/word2vec.pkl')
    
    train_d = Data('model/train_corpus.pkl', label_path, word_model, padding_length=80)#63
    train_loader = DataLoader(train_d, batch_size=64, shuffle=False)

    model = LSTMClassifier(embedding_dim=256, hidden_dim=hidden_dim).cuda()
    if is_save:
        model.load_state_dict(torch.load('model/model.mdl'))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        loss_list, acc_list = train(train_loader, model, optimizer, criterion, num_epochs)
        torch.save(model.state_dict(), 'model/model.mdl')
