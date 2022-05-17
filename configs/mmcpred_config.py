from pathlib import Path

BASE_DIR = Path('./')


class SmallConfig(object):
    """Small config."""
    mode = 'GRU'
    init_scale = 0.1
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 2
    num_gen = 3
    num_heads = 4
    num_blocks = 5
    in_channels = 1
    hidden_size = 200
    filter_output_dim = 200
    filter_size = 3
    padding_mode = 'replicate'
    stride = 1
    keep_prob = 1.0
    res_rate = 0.3
    lr_decay = 0.5
    batch_size = 100
    label_size = 10
    vocab_size = None


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 122911
    label_size = 20
    rnn_mode = 'block'


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    label_size = 20
    rnn_mode = 'block'


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    label_size = 20
    rnn_mode = 'block'


config = {
    'data_dir':  BASE_DIR / 'dataset' / 'MMCPred',
    'log_dir': BASE_DIR / 'output/log',
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'result_dir':  BASE_DIR / "output/result",
}
