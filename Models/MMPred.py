import torch
import torch.nn as nn

from .Components.encoders import RNNEncoder
from .Components.decoders import RNNDecoder


class InputLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(InputLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(in_features=self.input_size,
                                out_features=self.output_size)

    def forward(self, inputs):
        return torch.relu_(self.linear(inputs))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.mode = config.mode
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.keep_prob = config.keep_prob
        self.vocab_size = config.vocab_size
        self.num_tasks = len(self.vocab_size)
        self.hidden_size_vocab = config.hidden_size_attention
        self.init_scale = config.init_scale

        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.vocab_size[i],
                                                      embedding_dim=self.hidden_size_vocab,
                                                      max_norm=self.init_scale) for i in range(self.num_tasks)])

        self.inputlayers = nn.ModuleList([InputLayer(input_size=self.hidden_size_vocab,
                                                     output_size=self.hidden_size) for _ in range(self.num_tasks)])

        self.encoders = nn.ModuleList([RNNEncoder(mode=self.mode,
                                                  input_size=self.hidden_size,
                                                  hidden_size=self.hidden_size,
                                                  num_layers=self.num_layers,
                                                  dropout=1 - self.keep_prob) for _ in range(self.num_tasks)])

        self.modulaters = nn.ModuleList([nn.Linear(in_features=self.num_tasks * self.hidden_size * 2,
                                                   out_features=self.num_tasks) for _ in range(self.num_tasks)])

        # z = tf.concat([tf.multiply(outputs_e0[-1], outputs_e1[-1]), outputs_e0[-1], outputs_e1[-1]], 1)
        # with tf.variable_scope("modulator_e"):
        #     z_w = tf.get_variable("z_w", [z.get_shape()[1], 2], dtype=tf.float32)
        #     z_b = tf.get_variable("z_b", [2], dtype=tf.float32)
        #     logits_z = tf.nn.xw_plus_b(z, z_w, z_b)
        #     b = tf.sigmoid(logits_z)

    def forward(self, inputs_list):
        hidden_states_list = []
        hidden_last_list = []
        for i in range(self.num_tasks):
            inputs = self.embeddings[i](inputs_list[i])
            inputs = self.inputlayers[i](inputs)
            hidden_states = self.encoders[i](inputs)
            hidden_states_list.append(hidden_states.unsqueeze(1))
            hidden_last_list.append(hidden_states[:, -1, :].squeeze().unsqueeze(0))

        encoder_outputs_list = []
        for i in range(self.num_tasks):
            hidden_rz = hidden_last_list[i] * torch.cat(hidden_last_list, dim=0)
            rz = torch.cat([hidden_rz, torch.cat(hidden_last_list, dim=0)])
            modulate_weight = self.modulaters[i](rz.transpose(1, 0).reshape(-1, self.num_tasks * self.hidden_size * 2))
            modulate_weight = torch.sigmoid(modulate_weight)  # [B, num_tasks]
            all_hidden_states = torch.cat(hidden_states_list, dim=1)
            hidden_states = torch.mul(modulate_weight.unsqueeze(2).unsqueeze(3).expand_as(all_hidden_states),
                                      all_hidden_states)
            encoder_outputs_list.append(hidden_states.sum(dim=1).squeeze())

        return encoder_outputs_list


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.mode = config.mode
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.keep_prob = config.keep_prob
        self.vocab_size = config.vocab_size
        self.num_tasks = len(self.vocab_size)

        self.decoders = nn.ModuleList([RNNDecoder(mode=self.mode,
                                                  input_size=self.hidden_size,
                                                  hidden_size=self.hidden_size,
                                                  num_layers=self.num_layers,
                                                  dropout=1 - self.keep_prob) for _ in range(self.num_tasks)])

        self.linears = nn.ModuleList([nn.Linear(in_features=self.hidden_size,
                                                out_features=self.vocab_size[i],
                                                bias=True) for i in range(self.num_tasks)])

    def forward(self, encoder_outputs_list):
        logits_list = []
        for i in range(self.num_tasks):
            outputs = self.decoders[i](encoder_outputs_list[i])
            logits = self.linears[i](outputs)
            logits_list.append(logits)
        return logits_list


class MMPred(nn.Module):
    def __init__(self, config):
        super(MMPred, self).__init__()

        self.num_tasks = len(config.vocab_size)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, input_ids):
        """
        :param inputs_event_list: a batch of multiple sequence [B, num_tasks, ln]
        :return:
        """
        inputs_event_list = [input_ids[:, i, :] for i in range(input_ids.size(1))]
        encoder_outputs_list = self.encoder(inputs_event_list)
        logits_list = self.decoder(encoder_outputs_list)

        return logits_list
