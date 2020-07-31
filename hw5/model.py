from util import *



#Word Embedding Model 1
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.3, num_layers=4)
        self.hidden2out = nn.Linear(hidden_dim, 2)
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        
        tag_scores = self.hidden2out(lstm_out[-1].view(-1, self.hidden_dim))
        return tag_scores

# Word Embedding Model 2
class model_2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(model_2, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.5, num_layers=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_scores = self.fc(lstm_out[-1].view(-1, self.hidden_dim))
        return tag_scores

# Word Embedding Model 3
class model_3(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(model_3, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.3, num_layers=4)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.hidden2out = nn.Linear(hidden_dim*2, 2)
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        attn_out, _ = self.attn(lstm_out, sentence, sentence)
        attn_out = attn_out.mean(dim=0).view(-1, self.hidden_dim)
        x = torch.cat((lstm_out[-1].view(-1, self.hidden_dim), attn_out), dim=1)
        tag_scores = self.hidden2out(x)
        return tag_scores

# BOW Model 1
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            self.fc_layer(9358, 512),
            self.fc_layer(512, 512),
            self.fc_layer(512, 512),
            self.fc_layer(512, 2),
        )
    def fc_layer(self, in_dim, out_dim, dropout=0.3):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.fc(x)
        return x