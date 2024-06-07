import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.name = 'LSTM'
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out[:, -1, :]

        priority = self.fc(lstm_out)

        return priority

