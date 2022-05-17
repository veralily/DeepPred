import collections
import pandas as pd
import random
import torch
import logging
import json

from .utils import load_pickle, save_pickle
from .progressbar import ProgressBar
from torch.utils.data import TensorDataset


class InputExample(object):
    def __init__(self, guid, sequence):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            sequence: list of multi-attribute events list.
        """
        self.guid = guid
        self.sequence = sequence


class InputFeature(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids):
        self.input_ids = input_ids

    def value(self):
        result = []
        result += [self.input_ids]
        return result


class TaskData(object):
    def __init__(self, raw_data_path, num_tsaks=None, preprocessor=None, is_train=True, logger=None):
        self.data_path = raw_data_path
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.num_tasks = num_tsaks
        self.logger = logger if logger else logging.getLogger()

    def read_data(self, length, get_attribute_name=False):
        """
        :@param relations: list of str
        :return: list of targets and sentences
        """
        sequences_list = []

        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = [items.strip().split('-') for items in line.split(':')[1].split()]
                data = pd.DataFrame(line)
                if len(data) < length:
                    continue
                else:
                    num_cols = len(data.columns)
                    for row_index in range(len(data) - length):
                        sequence_data = data.loc[row_index: row_index+length-1, :]
                        sequences_list.append([sequence_data.loc[:, col_index].tolist() for col_index in range(num_cols)])

        self.num_tasks = num_cols
        return sequences_list

    def build_vocab(self, save_vocab=None):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            f.close()
        self.logger.info(f'the number of lines in {self.data_path}: {len(lines)}')
        line = lines[0].split(':')[1].split()[0]
        num_tasks = len(line.split('-'))
        self.logger.info(f'the number of tasks in {self.data_path}: {num_tasks}')
        data = [[] for _ in range(num_tasks)]
        for i, line in enumerate(lines):
            line = line.split(':')[1].strip()
            for item in line.split():
                words = item.split('-')
                for j in range(num_tasks):
                    data[j].append(words[j])

        word2Id = dict()
        for i in range(num_tasks):
            counter = collections.Counter(data[i])
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            words, _ = list(zip(*count_pairs))
            word_to_id = dict(zip(words, range(len(words))))
            word2Id[i] = word_to_id
        if save_vocab:
            if 'pkl' in str(save_vocab):
                self.logger.info(f'save data in pkl to {save_vocab}')
                save_pickle(data=word2Id, file_path=save_vocab)
            else:
                self.logger.info(f'save data in json to {save_vocab}')
                with open(save_vocab, 'w') as f:
                    f.write(json.dumps(word2Id,ensure_ascii=False, indent=4, separators=(',', ':')))
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
        self.logger.info(f'save data in pkl for {data_split}')
        data = X
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        file_path = data_dir / f"{data_name}.{data_split}.pkl"
        save_pickle(data=data, file_path=file_path)


class MMPredProcessor(object):
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
        return lines

    @classmethod
    def read_data(cls, input_file):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def tokenizer(self, sequence):
        if len(self.word2id) != len(sequence):
            sequence = sequence[:len(self.word2id)]
        assert len(self.word2id) == len(sequence)
        results = []
        for i, each_sequence in enumerate(sequence):
            word2id = self.word2id[str(i)]
            input_ids = [word2id[word] if word in word2id else word2id['UNK'] for word in each_sequence]
            results.append(torch.tensor(input_ids, dtype=torch.long).unsqueeze(0))
        return torch.cat(results, dim=0)

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
            for i, line in enumerate(lines):
                guid = '%s-%d' % (data_split, i)
                sequence = line
                example = InputExample(guid=guid, sequence=sequence)
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
                input_ids = self.tokenizer(example.sequence)  # tensors: [num_tasks, len]
                # print(example.target)

                if ex_id < 2:
                    self.logger.info("*** Example ***")
                    self.logger.info(f"guid: {example.guid}" % ())
                    self.logger.info(f"sequence: {example.sequence}")
                    self.logger.info(f"input_ids: {input_ids}")

                feature = InputFeature(input_ids=input_ids)
                features.append(feature)
                if (ex_id + 1) % print_interval == 0:
                    self.logger.info(pbar(step=ex_id))
            self.logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self, features):
        # Convert to Tensors and build dataset
        all_input_ids = torch.cat([f.input_ids.unsqueeze(0) for f in features], dim=0)
        # [num_examples, num_tasks, len]
        dataset = TensorDataset(all_input_ids)
        return dataset