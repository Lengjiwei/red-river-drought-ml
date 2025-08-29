# model.py — minimal CNN-LSTM skeleton (placeholder)
import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, in_channels=1, cnn_channels=(16,32), lstm_hidden=64, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels[1], lstm_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden, 1))
    def forward(self, x):
        # x: [B, T, F]; treat F as channels
        x = x.transpose(1, 2)       # [B, F, T]
        x = self.cnn(x).transpose(1, 2)
        out, _ = self.lstm(x)
        y = self.head(out[:, -1, :]).squeeze(-1)
        return y
