import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import os, sys
from process_bar import ShowProcess

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def output(y, path='./output.csv'):
    idx = np.arange(y.shape[0]).reshape(-1, 1)
    with open(path, 'w') as f:
        f.write(pd.DataFrame(np.c_[idx, y], columns=['id', 'label']).to_csv(index=False))

class Data(Dataset):
    def __init__(self, path):
        self.images = np.load(path)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.transpose(img, (2, 0, 1)) / 255. * 2 - 1
        img = torch.Tensor(img).cuda()
        return img

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, 3, 2, 1),
            nn.SELU(),         
            nn.Conv2d(4, 8, 3, 2, 1),
            nn.SELU(),            
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.SELU(),            
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SELU(),            
            nn.Conv2d(32, 64, 2, 1),
            nn.SELU(),            
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 1), 
            nn.SELU(),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.SELU(),
            nn.ConvTranspose2d(16, 8, 2, 2),
            nn.SELU(),
            nn.ConvTranspose2d(8, 4, 2, 2),
            nn.SELU(),
            nn.ConvTranspose2d(4, 3, 2, 2),
            nn.Tanh(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train(train_loader, model, optimizer, criterion, num_epochs, show=True):
    for epoch in range(num_epochs):
        if show:
            process_bar = ShowProcess(len(train_loader))
        loss_cnt = 0
        for i, d in enumerate(train_loader):
            latent, reconstruct = model(d)
            loss = criterion(reconstruct, d)
            loss_cnt += loss.item()* d.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [%03d/%03d], ' % (epoch+1, num_epochs), end='')
            if show:
                process_bar.show_process(other=', loss: %.4f' % (loss_cnt/len(train_d)))

if __name__ == '__main__':
    path = sys.argv[1]
    output_path = sys.argv[2]
    is_save = True
    
    batch_size = 2000
    lr = 1e-3
    num_epochs = 100
    train_d = Data(path)
    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_d, batch_size=batch_size, shuffle=False)

    img_encoder = Autoencoder().cuda()
    
    if is_save:
        img_encoder.load_state_dict(torch.load('autoencoder.model'))
    else:
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(img_encoder.parameters(), lr=lr, weight_decay=1e-5)
        train(train_loader, img_encoder, optimizer, criterion, num_epochs)
        torch.save(img_encoder.state_dict(), 'autoencoder.model')

    latents = []
    reconstructs = []
    for x in test_loader:
        latent, reconstruct = img_encoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / (np.std(latents, axis=0)+1e-40)

    decompo = PCA(n_components=16, whiten=True).fit_transform(latents)
    result = KMeans(n_clusters = 2).fit(decompo).labels_
    if np.sum(result[:5]) >= 3:
        result = 1 - result

    output(result, output_path)
