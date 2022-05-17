import collections
import random
import torch
from .utils import load_pickle, save_pickle
from .progressbar import ProgressBar
from torch.utils.data import TensorDataset
import logging

import json


class InputExample(object):
    def __init__(self, guid, event_sequence, time_sequence):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            event_sequence: list of event (string).
            time_sequence: list of time (float)
        """
        self.guid = guid
        self.event_sequence = event_sequence
        self.time_sequence = time_sequence


class InputFeature(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, input_times):
        self.input_ids = input_ids
        self.input_times = input_times

    def value(self):
        result = []
        result += [self.input_ids]
        result += [self.input_times]
        return result


class TaskData(object):
    def __init__(self, raw_data_path, preprocessor=None, is_train=True, logger=None):
        self.data_path = raw_data_path
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.logger = logger if logger else logging.getLogger()

    def read_data(self, length):
        """
        :@param relations: list of str
        :return: list of targets and sentences
        """
        sequences_list = []

        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = line.strip().split('\t')
                if len(data) < length:
                    continue
                else:
                    for row_index in range(len(data) - length):
                        sequence_data = data[row_index: row_index+length]
                        sequences_list.append(sequence_data)

        return sequences_list

    def build_vocab(self, save_vocab=None):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            f.close()
        self.logger.info(f'the number of lines in {self.data_path}: {len(lines)}')
        data = []
        for i, line in enumerate(lines):
            for item in line.strip().split('\t'):
                word = item.split('-')[0]
                data.append(word)

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word2Id = dict(zip(words, range(len(words))))
        if save_vocab:
            if 'pkl' in str(save_vocab):
                self.logger.info(f'save data in pkl to {save_vocab}')
                save_pickle(data=word2Id, file_path=save_vocab)
            else:
                self.logger.info(f'save data in json to {save_vocab}')
                with open(save_vocab, 'w') as f:
                    f.write(json.dumps(word2Id, ensure_ascii=False, indent=4, separators=(',', ':')))
        return word2Id

    def save_data(self, X, shuffle=True, seed=None, data_name=None, data_dir=None, data_split=None):
        """
        save data in pkl format
        :param data_split: train or valid or inference
        :param X: the truncated sequence
        :param shuffle:
        :param seed:
        :param data_name:
        :param data_dir:
        :return:
        """
        self.logger.info(f'save data from {self.data_path} in pkl for {data_split}')
        data = []
        for step, sequence in enumerate(X):
            data_x = [str(item).split('-')[0] for item in sequence]
            data_y = [float(str(item).split('-')[1]) for item in sequence]
            data.append((data_x, data_y))
        del X

        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        file_path = data_dir / f"{data_name}.{data_split}.pkl"
        save_pickle(data=data, file_path=file_path)


class MMCPredProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, word2id, logger=None):
        self.word2id = word2id
        self.logger = logger if logger else logging.getLogger()

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self, lines):
        data = []
        for step, sequence in enumerate(lines):
            data_x = [str(item).split('-')[0] for item in sequence]
            data_y = [float(str(item).split('-')[1]) for item in sequence]
            data.append((data_x, data_y))
        return data

    @classmethod
    def read_data(cls, input_file):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def tokenizer(self, sequence):
        input_ids = [self.word2id[word] if word in self.word2id else self.word2id['UNK'] for word in sequence]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def create_examples(self, lines, data_split, cached_examples_file):
        """
        Creates examples for data
        """
        pbar = ProgressBar(n_total=len(lines), desc='create examples')
        print_interval = max(1, len(lines) // 10)
        if cached_examples_file.exists():
            self.logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i, (event_sequence, time_sequence) in enumerate(lines):
                guid = '%s-%d' % (data_split, i)
                example = InputExample(guid=guid, event_sequence=event_sequence, time_sequence=time_sequence)
                examples.append(example)
                if (i + 1) % print_interval == 0:
                    self.logger.info(pbar(step=i))
            self.logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features(self, examples, cached_features_file):
        """
        Creates features for each example data
        """
        pbar = ProgressBar(n_total=len(examples), desc='create features')
        print_interval = max(1, len(examples) // 10)
        if cached_features_file.exists():
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id, example in enumerate(examples):
                input_ids = self.tokenizer(example.event_sequence)  # tensors: [len]
                input_times = torch.tensor(example.time_sequence)  # [len]

                if ex_id < 2:
                    self.logger.info("*** Example ***")
                    self.logger.info(f"guid: {example.guid}" % ())
                    self.logger.info(f"sequence: {example.event_sequence}")
                    self.logger.info(f"input_ids: {input_ids}")
                    self.logger.info(f"input_times: {example.time_sequence}")

                feature = InputFeature(input_ids=input_ids, input_times=input_times)
                features.append(feature)
                if (ex_id + 1) % print_interval == 0:
                    self.logger.info(pbar(step=ex_id))
            self.logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self, features):
        # Convert to Tensors and build dataset
        all_input_ids = torch.cat([f.input_ids.unsqueeze(0) for f in features], dim=0)
        all_input_times = torch.cat([f.input_times.unsqueeze(0) for f in features], dim=0)
        # [num_examples, num_tasks, len]
        dataset = TensorDataset(all_input_ids, all_input_times)
        return dataset