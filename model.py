# model.py

import torch
import torch.nn as nn

class ContextRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=2, rnn_type="vanilla"):
        """
        RNN model for context-dependent decision making.

        Args:
            input_size (int): Size of the input vector (stimulus + context)
            hidden_size (int): Number of hidden units
            output_size (int): Number of output classes (default 2 for binary)
            rnn_type (str): "vanilla" or "gru"
        """
        super().__init__()
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, seq_len, input_size)
        Returns:
            logits (Tensor): (batch_size, output_size)
            hidden_states (Tensor): (1, batch_size, hidden_size)
        """
        out, h = self.rnn(x)
        logits = self.classifier(out[:, -1, :])
        return logits, h
