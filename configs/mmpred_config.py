from pathlib import Path

BASE_DIR = Path('./')


class BasicConfig(object):
    """Small config."""
    mode = 'LSTM'
    batch_size = 100
    num_steps = 5
    num_tasks = 2
    num_layers = 2
    hidden_size_vocab = 60
    hidden_size = 100
    hidden_size_attention = 20
    init_scale = 0.05
    max_grad_norm = 5
    output_length = 5
    keep_prob = 0.8
    reg_scale = 1.0
    label_size = 10
    vocab_size = None


config = {
    'data_dir':  BASE_DIR / 'dataset' / 'MMPred',
    'log_dir': BASE_DIR / 'output/log',
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'result_dir':  BASE_DIR / "output/result",
}


