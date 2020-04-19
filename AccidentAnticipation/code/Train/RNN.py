# Author : Ketan Kokane <kk7471@rit.edu>

import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_layer_1 = nn.Linear(hidden_size, hidden_size)
        # self.hidden_layer_2 = nn.Linear(1000, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden))

        x = self.input_layer(combined)
        x = self.tanh(x) # pass through a activation function
        hidden = self.hidden_layer_1(x)
        # x = self.tanh(x) # pass through a activation function
        # hidden = self.hidden_layer_2(x)
        output = self.output_layer(hidden)
        sigmoid_op = self.sigmoid(output)
        return sigmoid_op, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)
