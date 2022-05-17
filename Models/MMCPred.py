import torch
import torch.nn as nn
import torch.nn.functional as F

from .Components.encoders import CNNEncoder, RNNEncoder
from .Components.decoders import RNNDecoder
from .Components.selector import Selector
from .Components.Operators.Transformer import MultiHeadAttention


class EventEncoder(nn.Module):
    def __init__(self, config):
        super(EventEncoder, self).__init__()
        self.mode = config.mode
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.keep_prob = config.keep_prob
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size)

        self.encoder = RNNEncoder(mode=self.mode,
                                  input_size=self.hidden_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  dropout=1.0 - self.keep_prob
                                  )

        self.multihead_attention = MultiHeadAttention(embed_dim=self.hidden_size,
                                                      num_heads=self.num_heads,
                                                      dropout=1.0-self.keep_prob,
                                                      is_decoder=False,
                                                      bias=True, )

    def forward(self, inputs_event):
        event_embeddings = self.embeddings(inputs_event)
        hidden_re = self.encoder(event_embeddings)
        hidden_re, _, _ = self.multihead_attention(hidden_re)

        return torch.mean(hidden_re, dim=1, keepdim=True)


class TimeEncoder(nn.Module):
    def __init__(self, config):
        super(TimeEncoder, self).__init__()
        self.mode = config.mode
        self.num_layers = config.num_layers
        self.num_blocks = config.num_blocks
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.filter_output_dim = config.filter_output_dim
        self.filter_size = config.filter_size
        self.stride = config.stride
        self.padding_mode = config.padding_mode
        self.num_heads = config.num_heads
        self.keep_prob = config.keep_prob
        self.res_rate = config.res_rate
        self.vocab_size = config.vocab_size

        self.encoder = CNNEncoder(num_blocks=self.num_blocks,
                                  in_channels=self.in_channels,
                                  out_channels=self.hidden_size,
                                  kernel_size=self.filter_size,
                                  res_rate=self.res_rate,
                                  num_layers=self.num_layers,
                                  stride=self.stride,
                                  padding_mode=self.padding_mode)

        self.multihead_attention = MultiHeadAttention(embed_dim=self.hidden_size,
                                                      num_heads=self.num_heads,
                                                      dropout=0.0,
                                                      is_decoder=False,
                                                      bias=True, )

    def forward(self, inputs_time):
        inputs_time = inputs_time.unsqueeze(1)
        hidden_rt = self.encoder(inputs_time)
        hidden_rt = torch.transpose(hidden_rt, 1, 2)
        hidden_rt, _, _ = self.multihead_attention(hidden_rt)
        return torch.mean(hidden_rt, dim=1, keepdim=True)


class EventDecoder(nn.Module):
    def __init__(self, config):
        super(EventDecoder, self).__init__()
        self.mode = config.mode
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.keep_prob = config.keep_prob
        self.learning_rate = config.learning_rate
        self.lr = config.learning_rate
        self.vocab_size = config.vocab_size

        self.decoder = RNNDecoder(mode=self.mode,
                                  input_size=self.hidden_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  dropout=1.0-self.keep_prob)
        self.linear = nn.Linear(in_features=self.hidden_size,
                                out_features=self.vocab_size)

    def forward(self, hidden_re, length):
        hidden_re = hidden_re.repeat([1, length, 1])
        outputs_e = self.decoder(hidden_re)
        logits_e = self.linear(outputs_e)
        return logits_e, outputs_e


class TimeDecoder(nn.Module):
    def __init__(self, config):
        super(TimeDecoder, self).__init__()
        self.mode = config.mode
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_gen = config.num_gen
        self.keep_prob = config.keep_prob
        self.vocab_size = config.vocab_size

        self.time_decoders = nn.ModuleList([RNNDecoder(mode=self.mode,
                                                       input_size=self.hidden_size * 2,
                                                       hidden_size=self.hidden_size,
                                                       num_layers=self.num_layers,
                                                       dropout=1-self.keep_prob) for _ in range(self.num_gen)])
        self.time_linears = nn.ModuleList([nn.Linear(in_features=self.hidden_size,
                                                     out_features=1) for _ in range(self.num_gen)])

    def forward(self, outputs_e, hidden_rt, length):
        # outputs_e = self.event_decoder(hidden_re)
        # logits_e = self.event_linear(outputs_e)
        hidden_rt = hidden_rt.repeat([1, length, 1])

        hidden_rt = torch.cat([hidden_rt, outputs_e], dim=-1)

        # all steps concat or last step of hidden states only
        # the input of each step of RNN decoder is depend on the form of hidden states (copy or list)
        pred_time_list = []
        for i in range(self.num_gen):
            outputs_t = self.time_decoders[i](hidden_rt)
            logits_t = self.time_linears[i](outputs_t)
            # [B, len, 1]
            pred_time_list.append(logits_t)
        return pred_time_list


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.num_layers = config.num_layers
        self.num_blocks = config.num_blocks
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.filter_output_dim = config.filter_output_dim
        self.filter_size = config.filter_size
        self.stride = config.stride
        self.padding_mode = config.padding_mode
        self.num_heads = config.num_heads
        self.keep_prob = config.keep_prob
        self.res_rate = config.res_rate
        self.vocab_size = config.vocab_size

        self.blocks = CNNEncoder(num_blocks=self.num_blocks,
                                 in_channels=self.in_channels,
                                 out_channels=self.hidden_size,
                                 kernel_size=self.filter_size,
                                 res_rate=self.res_rate,
                                 num_layers=self.num_layers,
                                 stride=self.stride,
                                 padding_mode=self.padding_mode)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1, bias=True)

    def forward(self, inputs):
        """
        :param inputs: [B, ln]
        :return: [B, 1]
        """
        hidden_states = self.blocks(inputs.unsqueeze(1))
        hidden_states = torch.transpose(hidden_states, 1, 2)
        scores = self.linear(hidden_states).squeeze(-1)  # [B, ln]
        return scores.sum()


class MMCPred(nn.Module):
    def __init__(self, config):
        super(MMCPred, self).__init__()
        self.num_gen = config.num_gen
        self.event_encoder = EventEncoder(config)
        self.time_encoder = TimeEncoder(config)
        self.event_decoder = EventDecoder(config)
        self.time_decoder = TimeDecoder(config)
        self.time_selector = Selector(in_features=config.hidden_size * 2,
                                      num_labels=config.num_gen)
        self.discriminator = Discriminator(config)

    def forward(self, inputs_event, inputs_time, length):
        hidden_re = self.event_encoder(inputs_event)
        hidden_rt = self.time_encoder(inputs_time)

        logits_e, outputs_e = self.event_decoder(hidden_re, length)
        list_logits_t = self.time_decoder(outputs_e, hidden_rt, length)

        time_decoder_weights = self.time_selector(torch.cat([hidden_re, hidden_rt], dim=-1))  # [B, num_gen]

        k = time_decoder_weights.argmax(dim=-1)
        outputs_t = torch.cat(list_logits_t, dim=-1)  # [B, ln, num_gen]
        index_choices = F.one_hot(k, num_classes=self.num_gen)
        index_choices = index_choices.repeat([1, length, 1]).bool()
        logits_t = outputs_t.masked_select(index_choices)
        logits_t = logits_t.unsqueeze(1) if length == 1 else logits_t

        weights = F.gumbel_softmax(time_decoder_weights)  # [B, 1, num_gen]
        gumbel_softmax_logits_t = torch.mul(weights.repeat([1, length, 1]), outputs_t).sum(dim=-1)

        return logits_e, logits_t, gumbel_softmax_logits_t
