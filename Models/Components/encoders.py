import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import torch.nn.functional as F
import math
import random
from transformers import BartConfig, BartPretrainedModel

from .Operators.Transformer import BartEncoderLayer
from .utils import BartLearnedPositionalEmbedding, expand_mask
from .Operators.CNN import CONV1D, ResBlock
from .Operators.RNN import RNNLayers


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super(BartEncoder, self).__init__(config)

        self.config = config

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return {
            "last_hidden_state": hidden_states, "hidden_states": encoder_states, "attentions": all_attentions}


class CNNEncoder(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size, res_rate, num_layers,
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(CNNEncoder, self).__init__()
        self.num_blocks = num_blocks

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.res_rate = res_rate
        self.num_layers = num_layers

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        self.input_layer = CONV1D(in_channels=self.in_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding,
                                  dilation=self.dilation,
                                  groups=self.groups,
                                  bias=self.bias,
                                  padding_mode=self.padding_mode)

        self.layers = nn.ModuleList([ResBlock(in_channels=self.out_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=self.kernel_size,
                                              res_rate=self.res_rate,
                                              num_layers=self.num_layers,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              groups=self.groups,
                                              bias=self.bias,
                                              padding_mode=self.padding_mode,
                                              ) for _ in range(num_blocks)])

    def forward(self, inputs):
        outputs = self.input_layer(inputs)
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs


class RNNEncoder(nn.Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = True,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0, ):
        super(RNNEncoder, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size

        self.layers = RNNLayers(mode=self.mode,
                                input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                bias=self.bias,
                                batch_first=self.batch_first,
                                dropout=self.dropout,
                                bidirectional=self.bidirectional,
                                proj_size=self.proj_size)

    def forward(self, inputs, hx=None):
        return self.layers(inputs, hx)
