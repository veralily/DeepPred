import torch
import torch.nn as nn

from .Operators.Transformer import Attention


class Selector(nn.Module):
    def __init__(self, in_features, num_labels):
        super(Selector, self).__init__()
        self.in_features = in_features
        self.num_labels = num_labels

        self.selector = nn.Linear(in_features=self.in_features,
                                  out_features=self.num_labels,
                                  bias=True)

    def forward(self, inputs):
        logits = self.selector(inputs)
        return logits.softmax(dim=-1)
