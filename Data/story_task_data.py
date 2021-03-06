import pandas as pd
import torch
from torch.utils.data import TensorDataset
import random

from .utils import save_pickle, load_pickle
from .progressbar import ProgressBar
import logging


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, target=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for inference examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.target = target


class InputFeature(object):
    '''
    A single set of features of data.
    '''

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, input_len, labels_mask, labels_len, cls_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len
        self.labels_mask = labels_mask
        self.labels_len = labels_len
        self.cls_label = cls_label

    def value(self):
        result = []
        result += [self.input_ids]
        result += [self.labels_mask]
        result += [self.labels_len]
        result += [self.cls_label]
        return result


class TaskData(object):
    def __init__(self, raw_data_path, preprocessor=None, is_train=True, logger=None):
        self.data_path = raw_data_path
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.logger = logger if logger else logging.getLogger()

    def read_data(self):
        """
        :return: list of targets and sentences
        targets: the same as stories,
        stories: five sentences,
        chars: five None token
        """
        self.logger.info(f'context_motiv_task read data from {self.data_path}')
        self.logger.info(f'is_train tag: {self.is_train}')
        targets, stories, chars = [], [], []
        data = pd.read_csv(self.data_path)
        length = len(data.values)
        self.logger.info(f'number of data in {self.data_path}: {length}')
        for i in range(length):
            context = str(data.loc[i, 'context'])
            context_list = context.split('\t')[1:]
            assert len(context_list) == 5
            if self.preprocessor:
                context_list = [self.preprocessor(context) for context in context_list]

            targets.append(context_list[1:])
            stories.append(context_list[:-1])
            chars.append([None] * 4)
        return targets, stories, chars

    def save_data(self, X, y, c=None, shuffle=False, seed=None, data_name=None, data_dir=None, data_split=None):
        """
        save data in pkl format
        :param c: List(str)
        :param data_split: train or valid or inference
        :param X: List(list(str))
        :param y: List(list(str))
        :param shuffle:
        :param seed:
        :param data_name:
        :param data_dir:
        :return:
        """
        pbar = ProgressBar(n_total=len(X), desc='save')
        self.logger.info(f'save data in pkl for {data_split}')
        self.logger.info(f'number of data: {len(X)}')
        if c:
            data = []
            for step, (data_x, data_y, data_c) in enumerate(zip(X, y, c)):
                data.append((data_x, data_y, data_c))
                pbar(step=step)
            del X, y, c
        else:
            data = []
            for step, (data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y, None))
                pbar(step=step)
            del X, y
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        file_path = data_dir / f"{data_name}.{data_split}.pkl"
        save_pickle(data=data, file_path=file_path)


def concat_tokens_pair(tokens_a, tokens_b, right_segment_ids, max_len, pad_token_id, seg_token_id):
    """
    tokens_a ids: (ln)
    """

    a_ids = tokens_a['input_ids'][0]
    b_ids = tokens_b['input_ids'][0]
    a_len = tokens_a['length'][0]
    b_len = tokens_b['length'][0]

    left_ids = a_ids
    right_ids = b_ids
    if a_len + b_len > max_len:
        print(f"a_len: {a_len}, b_len: {b_len}, max_len: {max_len}")
        left_ids = torch.cat([left_ids[:max_len - 1 - b_len], torch.tensor([seg_token_id], dtype=left_ids.dtype)])
        input_ids = torch.cat([left_ids, right_ids[:b_len]])
        if right_segment_ids is None:
            segment_ids = torch.cat([torch.zeros([max_len - b_len], dtype=torch.int),
                                     torch.ones([b_len], dtype=torch.int)])
        else:
            segment_ids = torch.cat([torch.zeros([max_len - b_len], dtype=torch.int),
                                     1 + right_segment_ids[:b_len]])
        input_masks = torch.ones([max_len], dtype=torch.int)
        input_lens = max_len
    else:
        input_ids = torch.cat([left_ids[:a_len], right_ids[:b_len]])
        input_len = a_len + b_len
        input_ids = torch.cat(
            [input_ids, pad_token_id * torch.ones([max_len - input_len], dtype=torch.int)])
        if right_segment_ids is None:
            segment_ids = torch.cat([torch.zeros([a_len], dtype=torch.int),
                                     torch.ones([b_len], dtype=torch.int),
                                     torch.zeros([max_len - input_len], dtype=torch.int)])
        else:
            segment_ids = torch.cat([torch.zeros([a_len], dtype=torch.int),
                                     1 + right_segment_ids[:b_len],
                                     torch.zeros([max_len - input_len], dtype=torch.int)])
        input_masks = torch.cat([torch.ones([input_len], dtype=torch.int),
                                 torch.zeros([max_len - input_len], dtype=torch.int)])
        input_lens = input_len
    return {"input_ids": [input_ids],
            "segment_ids": [segment_ids],
            "attention_masks": [input_masks],
            "length": [input_lens]}


def truncate_ids(tokens, max_len, pad_token_id):
    ids = tokens['input_ids'][0]
    masks = tokens['attention_mask'][0]
    lens = tokens['length'][0]

    if lens >= max_len:
        input_ids = ids[:max_len]
        input_masks = masks[:max_len]
        input_lens = max_len
    else:
        input_ids = torch.cat([ids, torch.ones([max_len - lens], dtype=torch.int) * pad_token_id], dim=-1)
        input_masks = torch.cat([torch.ones([lens], dtype=torch.int),
                                 torch.zeros([max_len - lens], dtype=torch.int)],
                                dim=-1)
        input_lens = lens
    return input_ids, input_masks, input_lens


def create_dense_feature(sentence_list, tokenizer, max_seq_len, max_decode_len, his_len):
    """
    type of following variables are string
    sentence_list = [h, f, e, c, l]
    """
    history_sents = sentence_list[0]
    event = sentence_list[2]
    target = sentence_list[-1]

    tokens_history = tokenizer([history_sents], padding=True, return_length=True, return_tensors='pt')
    tokens_event = tokenizer([event], padding=True, return_length=True, return_tensors='pt')  # event sentence
    tokens_context = concat_tokens_pair(tokens_history,
                                        tokens_event,
                                        right_segment_ids=None,
                                        max_len=his_len + max_seq_len,
                                        pad_token_id=tokenizer.pad_token_id,
                                        seg_token_id=tokenizer.eos_token_id)

    # the type of following variables are list, flat them
    input_ids = tokens_context["input_ids"][0]
    input_masks = tokens_context["attention_masks"][0]
    segment_ids = tokens_context["segment_ids"][0]
    input_lengths = tokens_context["length"][0]

    tokens_target = tokenizer([target], padding=True, return_length=True, return_tensors='pt')
    labels_ids, labels_mask, labels_lens = truncate_ids(tokens_target,
                                                        max_len=max_decode_len,
                                                        pad_token_id=tokenizer.pad_token_id)

    assert labels_ids.size()[0] == max_decode_len
    assert labels_mask.size()[0] == max_decode_len

    return {"input_ids": input_ids,
            "input_masks": input_masks,
            "segment_ids": segment_ids,
            "input_lengths": input_lengths,
            "labels_ids": labels_ids,
            "labels_mask": labels_mask}


class BartProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, tokenizer, logger=None):
        self.tokenizer = tokenizer
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
    def read_data(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def create_examples(self, lines, data_split, cached_examples_file):
        """Creates examples for data"""
        self.logger.info(f"Number of data: {len(lines)}")
        pbar = ProgressBar(n_total=len(lines), desc='create examples')
        print_interval = max(1, len(lines) // 10)
        if cached_examples_file.exists():
            self.logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i, line in enumerate(lines):
                guid = '%s-%d' % (data_split, i)
                text_a = line[0]
                target = line[1]
                text_b = line[2]
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, target=target)
                examples.append(example)
                if (i + 1) % print_interval == 0:
                    self.logger.info(pbar(step=i))
            self.logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_fake_examples(self, lines, data_split, cached_examples_file):
        """Creates examples for data"""
        self.logger.info(f"Number of data: {len(lines)}")
        pbar = ProgressBar(n_total=len(lines), desc='create examples')
        print_interval = max(1, len(lines) // 10)
        if cached_examples_file.exists():
            self.logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            self.logger.info("Creating examples to cached file %s", cached_examples_file)
            examples = []

            all_targets = []
            for line in lines:
                target = line[1]
                all_targets += target
            length = len(all_targets)

            for i, line in enumerate(lines):
                guid = '%s-%d' % (data_split + "_fake", i)
                text_a = line[0]
                target = line[1]
                text_b = line[2]

                fake_targets = []
                for line_num, real_target in enumerate(target):
                    assert real_target is not None
                    while True:
                        randn = random.randint(0, length - 1)
                        fake_target = all_targets[randn]
                        if fake_target != real_target:
                            fake_targets.append(fake_target)
                            break
                assert len(fake_targets) == len(text_a)
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, target=fake_targets)
                examples.append(example)
                if (i + 1) % print_interval == 0:
                    self.logger.info(pbar(step=i))
            self.logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features(self,
                        examples,
                        max_seq_len,
                        max_decode_len,
                        cached_features_file,
                        sep=None,
                        his_len=100,
                        example_type=None):
        """
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0"""

        sep = sep if sep is not None else "</s>"

        pbar = ProgressBar(n_total=len(examples), desc='create features')
        print_interval = max(1, len(examples) // 10)
        if cached_features_file.exists():
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id, example in enumerate(examples):
                history_sents = ['<mask>' if i == 0 else sep.join(example.text_a[:i]) for i in range(len(example.text_a))]
                future_sents = [sep.join(example.text_a[i:]) for i in range(len(example.text_a))]
                event_sents = example.text_a
                char_sents = example.text_b
                label_sents = example.target

                for h, f, e, c, l in zip(history_sents, future_sents, event_sents, char_sents, label_sents):
                    sentence_list = [h, f, e, c, l]

                    # input prefix + event
                    feature_dict = create_dense_feature(sentence_list,
                                                        self.tokenizer,
                                                        max_seq_len,
                                                        max_decode_len,
                                                        his_len=his_len)

                    # print(example.target)
                    input_ids = feature_dict["input_ids"]
                    input_masks = feature_dict["input_masks"]
                    segment_ids = feature_dict["segment_ids"]
                    input_lens = feature_dict["input_lengths"]
                    labels_ids = feature_dict["labels_ids"]
                    labels_mask = feature_dict["labels_mask"]
                    if example_type is None:
                        label = 0
                    else:
                        assert example_type == "fake"
                        label = 1

                    if ex_id < 2:
                        self.logger.info("*** Example ***")
                        self.logger.info(f"guid: {example.guid}")
                        self.logger.info(f"tokens: {e}")
                        self.logger.info(f"input_ids: {input_ids.detach().cpu()}")
                        self.logger.info(f"input_mask: {input_masks.detach().cpu()}")
                        self.logger.info(f"segment_ids: {segment_ids.detach().cpu()}")
                        self.logger.info(f"label_ids: {labels_ids.detach().cpu()}")
                        self.logger.info(f"label_sentence: {l}")
                        self.logger.info(f"label: {label}")

                    feature = InputFeature(input_ids=input_ids,
                                           input_mask=input_masks,
                                           segment_ids=segment_ids,
                                           label_ids=labels_ids,
                                           input_len=input_lens,
                                           labels_mask=labels_mask,
                                           labels_len=sum(labels_mask),
                                           cls_label=label)
                    # if ex_id < 2:
                    #     logger.info(f"-------------------feature: {feature.value()}")
                    features.append(feature)
                if (ex_id + 1) % print_interval == 0:
                    self.logger.info(pbar(step=ex_id))
            self.logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self, features, is_sorted=False):
        # Convert to Tensors and build dataset
        if is_sorted:
            self.logger.info("sorted data by the length of input")
            features = sorted(features, key=lambda x: x.input_len, reverse=True)
        all_input_ids = torch.cat([f.input_ids.unsqueeze(0) for f in features], dim=0)
        all_input_mask = torch.cat([f.input_mask.unsqueeze(0) for f in features], dim=0)
        all_segment_ids = torch.cat([f.segment_ids.unsqueeze(0) for f in features], dim=0)
        all_label_ids = torch.cat([f.label_ids.unsqueeze(0) for f in features], dim=0)
        all_input_lens = torch.cat([f.input_len.unsqueeze(0) for f in features], dim=0)
        all_labels_mask = torch.cat([f.labels_mask.unsqueeze(0) for f in features], dim=0)
        all_labels_len = torch.cat([f.labels_len.unsqueeze(0) for f in features], dim=0)
        all_cls_label = torch.cat([torch.tensor(f.cls_label).unsqueeze(0) for f in features], dim=0)
        dataset = TensorDataset(all_input_ids,
                                all_input_mask,
                                all_segment_ids,
                                all_label_ids,
                                all_input_lens,
                                all_labels_mask,
                                all_labels_len,
                                all_cls_label)
        return dataset
