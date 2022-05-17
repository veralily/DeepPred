import torch
import time
import json
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.optim import SGD, AdamW
from .src.tools import init_logger, logger, seed_everything, load_pickle
from .src.callback.modelcheckpoint import ModelCheckpoint
from .src.callback.trainingmonitor import TrainingMonitor
from .src.callback.lr_schedulers import get_linear_schedule_with_warmup

from .src.train.metrics import Accuracy, Precision

from .configs.mmpred_config import BasicConfig, config
from .Data.utils import collate_fn_mmpred as collate_fn
from .Data.mmpred_task_data import TaskData, MMPredProcessor
from .Models.MMPred import MMPred
from .src.train.trainer_mmpred import Trainer


def build_vocab(data_name):
    data = TaskData(raw_data_path=config['data_dir'] / f"{data_name}/{data_name}.txt")
    data.build_vocab(save_vocab=config['data_dir'] / f"{data_name}/{data_name}_vocab.json")


def run_data(data_name, data_split, num_steps):
    is_train = True if data_split == 'train' else False
    data = TaskData(raw_data_path=config['data_dir'] / f"{data_name}/{data_name}_{data_split}.txt",
                    is_train=is_train)
    sequences = data.read_data(length=num_steps + 1)
    data.save_data(X=sequences,
                   shuffle=True,
                   data_name=data_name,
                   data_dir=config['data_dir'] / f"{data_name}",
                   data_split=data_split)

    build_vocab(data_name)


def get_features(data_name, processor, data, data_split):
    examples = processor.create_examples(
        lines=data,
        data_split=data_split,
        cached_examples_file=config['data_dir'] / f"{data_name}/cached_examples_{data_split}")
    features = processor.create_features(
        examples=examples,
        cached_features_file=config['data_dir'] / f"{data_name}/cached_features_{data_split}")
    return features


def load_train_valid_data(data_name, batch_size, word2id, sorted=True):
    processor = MMPredProcessor(word2id)
    train_data = processor.get_train(config['data_dir'] / f"{data_name}/{data_name}.train.pkl")
    valid_data = processor.get_dev(config['data_dir'] / f"{data_name}/{data_name}.valid.pkl")
    train_features = get_features(data_name, processor, train_data, "train")
    valid_features = get_features(data_name, processor, valid_data, "valid")

    train_features = train_features
    train_dataset = processor.create_dataset(train_features)

    if sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,
                                  collate_fn=collate_fn)

    valid_features = valid_features
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size,
                                  collate_fn=collate_fn)

    logger.info("***** Running training: Loading data *****")
    logger.info("  Train examples = %d", len(train_features))
    logger.info("  Valid examples = %d", len(valid_features))
    return train_dataloader, valid_dataloader


def run_train(args):

    # word2id = load_pickle(config['data_dir'] / f"{args.data_name}/{args.data_name}_vocab.pkl")
    with open(config['data_dir'] / f"{args.data_name}/{args.data_name}_vocab.json", 'r') as f:
        word2id = json.load(f)

    vocab_size = [len(i) for i in word2id.values()]
    BasicConfig.vocab_size = vocab_size
    logger.info(f'the vocab_size : {vocab_size}')

    # ------- data
    train_dataloader, valid_dataloader = load_train_valid_data(args.data_name, args.batch_size, word2id)
    # ------- model
    logger.info("initializing model")
    model = MMPred(config=BasicConfig)

    if args.resume_path:
        resume_path = f"{config['checkpoint_dir']}/{args.resume_path}/epoch_{args.resume_epoch}_mmpred_model.bin"
        states = torch.load(resume_path)
        logger.info(f"Load model from {args.resume_path} with epoch {states['epoch']}")
        model.load_state_dict(states["model"])
    else:
        resume_path = None

    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    warmup_steps = int(t_total * args.warmup_proportion)

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'] / args.loginfo, arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=f'{config["checkpoint_dir"]}/{args.arch}_{args.data_name}_{args.loginfo}',
                                       mode=args.mode,
                                       monitor=args.monitor, arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training: Training args *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    num_tasks = len(word2id)
    
    trainer = Trainer(args=args,
                      model=model,
                      logger=logger, optimizer=optimizer, scheduler=scheduler,
                      early_stopping=None, training_monitor=train_monitor, model_checkpoint=model_checkpoint,
                      batch_metrics_list=[[Accuracy(topK=1)] for _ in range(num_tasks)],
                      epoch_metrics_list=[[Precision()] for _ in range(num_tasks)],
                      resume_path=resume_path)

    # see the resource of trainer to find the progress of metric calculation!!!
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='mmpred', type=str)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument('--resume_epoch', default=15, type=int)
    parser.add_argument('--data_name', default='v4_atomic', type=str)
    parser.add_argument("--loginfo", default='', type=str)

    parser.add_argument("--do_data", action='store_true')  # for v4_atomic, change the relations for need
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_steps", default=5, type=int)

    parser.add_argument("--learning_rate", default=0.2, type=float)
    parser.add_argument("--weight_decay", default=1.0, type=float)
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--n_gpu", type=str, default='1', help='"0,1,.." or "0" or "" ')
    parser.add_argument("--sorted", default=0, type=int, help='1 : True  0:False ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    init_logger(
        log_file=f'{config["log_dir"]}/{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}-{args.loginfo}.log')
    import os
    checkpoint_dir = f'{config["checkpoint_dir"]}/{args.arch}_{args.data_name}_{args.loginfo}'
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(args, f'{checkpoint_dir}/training_args.bin')
    seed_everything(args.seed)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_data:
        run_data(args.data_name, 'train', args.num_steps)
        run_data(args.data_name, 'valid', args.num_steps)

    if args.do_train:
        run_train(args)

