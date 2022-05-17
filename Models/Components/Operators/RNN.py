import torch
import torch.nn as nn


class RNNLayers(nn.Module):
    """
    input_size – The number of expected features in the input x
    hidden_size – The number of features in the hidden state h
    num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form
    a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
    bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq,
    batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for
    details. Default: False
    dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with
    dropout probability equal to dropout. Default: 0
    bidirectional – If True, becomes a bidirectional LSTM. Default: False
    proj_size – If > 0, will use LSTM with projections of corresponding size. Default: 0
    """
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = True,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0,
                 device=None, dtype=None):
        super(RNNLayers, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size

        if mode == "LSTM":
            self.layers = nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bias=self.bias,
                                  batch_first=self.batch_first,
                                  dropout=self.dropout,
                                  bidirectional=self.bidirectional,)
        elif mode == "GRU":
            self.layers = nn.GRU(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 bias=self.bias,
                                 batch_first=self.batch_first,
                                 dropout=self.dropout,
                                 bidirectional=self.bidirectional)
        else:
            self.layers = nn.RNN(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 bias=self.bias,
                                 batch_first=self.batch_first,
                                 dropout=self.dropout,
                                 bidirectional=self.bidirectional)

    def forward(self, inputs, hx=None):
        output, _ = self.layers(inputs, hx)
        return output


