import torch
import torch.nn as nn

from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 model_type="RNN",
                 dropout=0.2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.model_type = model_type
        self.dropout = dropout

        self.rnn = getattr(nn, model_type)(input_size,
                                           hidden_size,
                                           num_layers=num_layers,
                                           dropout=dropout,
                                           bidirectional=bidirectional)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))