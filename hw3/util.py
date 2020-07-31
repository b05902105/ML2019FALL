import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import pandas as pd
from PIL import Image

import os, glob, sys
from os.path import join

from process_bar import ShowProcess

class face_dataset(Dataset):
    def __init__(self, path, label_path=None, is_train=True, idx=None, transform=None):
        self.label = None
        self.is_train = is_train
        if is_train:
            assert label_path, "without label path"
            self.label = pd.read_csv(label_path, header=0).to_numpy()[:, 1]
            # for validation (select the specified index)
            if type(idx) is np.ndarray:
                self.label = self.label[idx]
        self.images = np.sort(np.array(glob.glob(join(path, '*.jpg'))))
        # for validation (select the specified index)
        if type(idx) is np.ndarray:
            self.images = self.images[idx]
        
        self.transform = transform

    def __len__(self):
        return len(self.images)
    def get_label(self):
        return self.label.astype(np.int)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = Image.open(self.images[idx])
        img = np.array(img)
        if self.transform:
            img = self.transform(img)
        if self.is_train:
            label = self.label[idx]
            return img, label
        return img

def train(train_loader, model, optimizer, criterion, num_epochs, val_loader=None, show=True):
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(num_epochs):
        if show:
            process_bar = ShowProcess(len(train_loader))
        loss_cnt = 0
        acc_cnt = 0
        model.train()
        for i, d in enumerate(train_loader):
            img, label = d
            img = Variable(img).cuda()
            label = label.detach().cuda()
            output = model(img)
            
            loss = criterion(output, label)
            loss_cnt += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict = torch.argmax(output.data, dim=1)
            correct = (predict == label).sum().item()
            acc_cnt += correct / label.size(0)
            print('Epoch [%03d/%03d], ' % (epoch+1, num_epochs), end='')
            if show:
                process_bar.show_process(other=', acc: %.2f' % (acc_cnt/(i+1)))
            else:
                print('Step [%02d/%02d], loss: %.4f, acc: %.2f' % (i+1, len(train_loader), loss_cnt/(i+1), acc_cnt/(i+1)), '\r', end='')
        loss_list.append(loss_cnt/len(train_loader))
        acc_list.append(acc_cnt/len(train_loader))
        
        if val_loader:
            model.eval()
            for d in val_loader:
                with torch.no_grad():
                    img, label = d
                    img = Variable(img).cuda()
                    label = label.detach().cuda()
                    output = model(img)
                    
                    val_loss = criterion(output, label)
                    val_loss_list.append(val_loss.item())
                    
                    predict = torch.argmax(output.data, dim=1)
                    correct = (predict == label).sum().item()
                    val_acc_list.append(correct / label.size(0))
                print('val acc: %.2f' % (val_acc_list[-1]), '\r', end='')
        print()
    if val_loader:
        return loss_list, acc_list, val_loss_list, val_acc_list
    return loss_list, acc_list
        
def predict(model, test_loader, show=False):
    pred = np.zeros((1,1))
    model.eval()
    with torch.no_grad():
        if show:
            process_bar = ShowProcess(len(test_loader), verbose=1)
        for img in test_loader:
            if type(img) == list:
                img, _ = img
            out = model(img.cuda())
            out = out.argmax(dim=1, keepdim=True).to('cpu').numpy()
            pred = np.r_[pred, out]
            if show:
                process_bar.show_process()
    return pred.reshape(-1).astype(np.int)[1:]

def plot_image(p, factor=3, interpolation='lanczos'):
    plt.figure(figsize=(len(p)+factor, len(p)+factor))
    if torch.is_tensor(p):
        p = p[0]
    plt.imshow(p, cmap='gray', interpolation=interpolation)
    
def output(y, path='./output.csv'):
    idx = np.arange(y.shape[0]).reshape(-1, 1)
    with open(path, 'w') as f:
        f.write(pd.DataFrame(np.c_[idx, y], columns=['id', 'label']).to_csv(index=False))
        
def params_num(model):
    return sum([np.prod(list(p.size())) for p in model.parameters()])
