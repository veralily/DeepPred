import torch
import torch.nn as nn
import time, json
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from .Data.utils import collate_fn_mmcpred as collate_fn
from .Data.mmcpred_task_data import TaskData, MMCPredProcessor
from .src.tools import init_logger
from .configs.mmcpred_config import config, SmallConfig
from .inference.generator_mmcpred import Generator

from .Models.MMCPred import MMCPred


class MMCPredTest(nn.Module):
    def __init__(self, logger):
        super(MMCPredTest, self).__init__()
        self.logger = logger
        self.model_config = SmallConfig
        
    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            word2id = json.load(f)
        self.model_config.vocab_size = len(word2id.values())
        return word2id

    def get_test_data(self, file_path, word2id, example_path, cached_features_path, batch_size, num_steps):
        data = TaskData(raw_data_path=file_path,
                        is_train=False)
        lines = data.read_data(length=num_steps)

        print(f"num of lines: {len(lines)}")

        processor = MMCPredProcessor(word2id)

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

    def run_test(self, model_path, test_dataloader, input_length, output_length, device=None, save_path=None, topK=10):

        map_location = "cpu" if device is None else f"cuda:{device}"

        states = torch.load(model_path, map_location=map_location)
        self.logger.info(f"Load model from {model_path}")
        model = MMCPred(self.model_config)
        model.load_state_dict(states["model"])

        self.logger.info('model predicting....')

        generator = Generator(model=model, logger=self.logger, n_gpu=device)
        event_ids, target_ids, times, target_times = generator.predictor(test_dataloader,
                                                                         input_length,
                                                                         output_length,
                                                                         topK=topK)
        event_ids = event_ids.squeeze()
        print(f"event_ids: {event_ids.size()}\n target_ids: {target_ids.size()}")
        print(f"times: {times.size()}\n target_times: {target_times.size()}")
        
        if save_path:
            columns = ["pred-time", "label-event", "label-time"] + [f"pred-event-{i}" for i in range(topK)]
            data = pd.DataFrame(columns=columns)
            for i in range(topK):
                data[f"pred-event-{i}"] = event_ids[:, i].contiguous().view(-1).tolist()
            data[f"label-event"] = target_ids.contiguous().view(-1).tolist()
            data[f"pred-time"] = times.contiguous().view(-1).tolist()
            data[f"label-time"] = target_times.contiguous().view(-1).tolist()
            data.to_csv(save_path)
            self.logger.info(f"result saved to {save_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='mmcpred', type=str)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--resume_epoch", type=int, default=4)
    parser.add_argument('--data_name', default='event_story', type=str)
    parser.add_argument("--log_info", default="", type=str)

    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument("--input_length", default=20, type=int)
    parser.add_argument("--output_length", default=1, type=int)
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

    test = MMCPredTest(logger)
    test_data = test.get_test_data(file_path, vocab_path, example_path, cached_features_path, args.batch_size,
                                   args.input_length + args.output_length)
    test.run_test(model_path, test_data, args.input_length, args.output_length, args.n_gpu,
                  save_path=config['result_dir'] / f"{args.data_name}_result.csv")


if __name__ == '__main__':
    main()
