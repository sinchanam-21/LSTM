import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        # If bidirectional, the output of LSTM will be 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch size, sent len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch size, sent len, emb dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output: [batch size, sent len, hid dim * num directions]
        # hidden: [num layers * num directions, batch size, hid dim]
        # cell: [num layers * num directions, batch size, hid dim]
        
        # Concat the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        # hidden: [batch size, hid dim * num directions]
            
        return self.fc(hidden)
