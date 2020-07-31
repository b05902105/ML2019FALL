from torch import nn

class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2),  # b, 16, 10, 10
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            
            nn.Conv2d(64, 128, 5, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            
            nn.Conv2d(256, 256, 3, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 7),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return x
    
class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(512, 512, 3, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 7),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return x
    
class model_4(nn.Module):
    def __init__(self):
        super(model_4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, 3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, 7),
            #nn.BatchNorm1d(7),
            #nn.ReLU(True),
            
            #nn.Linear(512, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return x

class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, 3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 7),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return x