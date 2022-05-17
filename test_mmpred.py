import torch
import torch.nn as nn
import time, json
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from .Data.utils import collate_fn_mmpred as collate_fn
from .Data.mmpred_task_data import TaskData, MMPredProcessor
from .src.tools import init_logger
from .configs.mmpred_config import config, BasicConfig
from .inference.generator_mmpred import Generator
from .Models.MMPred import MMPred


class MMPredTest(nn.Module):
    def __init__(self, logger):
        super(MMPredTest, self).__init__()
        self.logger = logger
        self.mode_config = BasicConfig

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            word2id = json.load(f)
        self.mode_config.vocab_size = [len(i) for i in word2id.values()]
        return word2id

    def get_test_data(self, file_path, word2id, example_path, cached_features_path, batch_size, num_steps):
        data = TaskData(raw_data_path=file_path,
                        is_train=False,
                        logger=self.logger)
        lines = data.read_data(length=num_steps + 1)

        print(f"num of lines: {len(lines)}")
        processor = MMPredProcessor(word2id)
        test_data = processor.get_test(lines=lines)
        test_examples = processor.create_examples(
            lines=test_data,
            data_split='inference',
            cached_examples_file=example_path)
        test_features = processor.create_features(
            examples=test_examples,
            cached_features_file=cached_features_path, )
        test_features = test_features

        print(f"num of examples: {len(test_features)}")
        test_dataset = processor.create_dataset(test_features)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size,
                                     collate_fn=collate_fn)
        return test_dataloader

    def load_model(self, model_path, device=None):
        map_location = "cpu" if device is None else f"cuda:{device}"

        states = torch.load(model_path, map_location=map_location)
        self.logger.info(f"Load model from {model_path}")
        model = MMPred(self.mode_config)
        model.load_state_dict(states["model"])
        return model

    def run_test(self, model, test_dataloader, device=None, save_path=None):
        self.logger.info('model predicting....')

        generator = Generator(model=model, logger=self.logger, n_gpu=device)
        ids, targets = generator.predictor(data=test_dataloader)
        if save_path:
            num_tasks = ids.size(0)
            columns = [f"pred-{i}" for i in range(num_tasks)] + [f"label-{i}" for i in range(num_tasks)]
            data = pd.DataFrame(columns=columns)
            for i in range(num_tasks):
                data[f"pred-{i}"] = ids[i].contiguous().view(-1).tolist()
                data[f"label-{i}"] = targets[i].contiguous().view(-1).tolist()
            data.to_csv(save_path)
            self.logger.info(f"result saved to {save_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='mmpred', type=str)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--resume_epoch", type=int, default=4)
    parser.add_argument('--data_name', default='event_story', type=str)
    parser.add_argument("--log_info", default="", type=str)

    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--n_gpu", type=str, default='', help='"0,1,.." or "0" or "" ')

    args = parser.parse_args()

    logger = init_logger(
        log_file=config['log_dir'] / f'{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}-{args.log_info}.log')
    checkpoint_dir = config['checkpoint_dir'] / args.arch
    checkpoint_dir.mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'testing_args.bin')
    logger.info("Testing parameters %s", args)

    file_path = config['data_dir'] / f"{args.data_name}/{args.data_name}_test.txt"
    vocab_path = config['data_dir'] / f"{args.data_name}/{args.data_name}_vocab.json"
    example_path = config['data_dir'] / f"{args.data_name}/cached_test_examples"
    cached_features_path = config['data_dir'] / f"{args.data_name}/cached_test_features_{args.num_steps}"

    model_path = f'{args.resume_path}/epoch_{args.resume_epoch}_{args.arch}_model.bin'

    test = MMPredTest()
    test_data = test.get_test_data(file_path, vocab_path, example_path, cached_features_path, args.batch_size, args.num_steps)
    test.run_test(model_path, test_data, args.n_gpu, save_path=config['result_dir'] / f"{args.data_name}_result.csv")


if __name__ == '__main__':
    main()
