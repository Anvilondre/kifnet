import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64, n_layers=3, dropout=0.5):
        super(Encoder, self).__init__()

        self.n_features = n_features
        self.hidden_dim = embedding_dim
        self.num_layers = n_layers
        self.rnn = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=self.num_layers,
          batch_first=True,
          dropout=dropout
        )
    
    def forward(self, x):
        
        h_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        
        c_1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        
        outputs, (hidden, cell) = self.rnn(x, (h_1, c_1))
        
        return outputs, hidden , cell 
    

class LSTM_Single(nn.Module):

    def __init__(self, inp_dim, out_dim, embedding_dim=64, n_lstm_layers=3, lstm_dropout=0.5):
        super(LSTM_Single, self).__init__()
        
        self.encoder = Encoder(inp_dim, embedding_dim, n_lstm_layers, lstm_dropout)
        self.fc = nn.Linear(embedding_dim, out_dim)
        
    def forward(self, X):
        _, hidden, _ = self.encoder(X)
        hidden = hidden[-1, :].squeeze()
        
        out = self.fc(hidden)
        
        return out