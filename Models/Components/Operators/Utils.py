import torch.nn.functional as F
import torch
import math
from typing import Optional


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


def linear_act(x):
    return x


ACT2FN = {
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
}

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

