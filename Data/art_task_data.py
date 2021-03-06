import pandas as pd
import torch
import random
from torch.utils.data import TensorDataset

from .utils import save_pickle
from .utils import load_pickle
from .progressbar import ProgressBar
import logging


class TaskData(object):
    def __init__(self, raw_data_path, preprocessor=None, is_train=True, logger=None):
        self.data_path = raw_data_path
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.logger = logger if logger else logging.getLogger()

    def read_data(self):
        """
        :@param relations: list of str
        :return: list of targets and sentences
        """
        data = pd.read_csv(self.data_path)

        targets_list = data["ob2"].values
        sentences_list = data["ob1"].values
        relations_list = data["hyp"].values
        return list(targets_list), list(sentences_list), list(relations_list)

    def save_data(self, X, y, c=None, shuffle=True, seed=None, data_name=None, data_dir=None, data_split=None):
        """
        save data in pkl format
        :param data_split: train or valid or inference
        :param X:
        :param y:
        :param shuffle:
        :param seed:
        :param data_name:
        :param data_dir:
        :return:
        """
        pbar = ProgressBar(n_total=len(X), desc='save')
        self.logger.info('save data in pkl for training')
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


class BartProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        tokens_a = tokens_a.squeeze().tolist()
        tokens_b = tokens_b.squeeze().tolist()
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def create_examples(self, lines, data_split, cached_examples_file):
        '''
        Creates examples for data
        '''
        pbar = ProgressBar(n_total=len(lines), desc='create examples')
        print_interval = max(1, len(lines) // 10)
        if cached_examples_file.exists():
            self.logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i, line in enumerate(lines):
                guid = '%s-%d' % (data_split, i)
                text_a = line[0].replace("<blank>", "<mask>")
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
        '''
        Creates examples for data
        '''
        pbar = ProgressBar(n_total=len(lines), desc='create examples')
        print_interval = max(1, len(lines) // 10)
        if cached_examples_file.exists():
            self.logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            import pandas as pd
            import random
            data = pd.DataFrame(lines, columns=["text_a", "target", "text_b"])
            realtions = set(data["text_b"].values)

            index = 0
            for relation in realtions:
                cur_data = data.loc[(data["text_b"] == relation)]
                for i in range(len(cur_data)):
                    index += 1
                    guid = '%s-%d' % (data_split + "_fake", index)
                    text_a = cur_data.iloc[i]["text_a"]
                    target = cur_data.iloc[i]["target"]
                    text_b = cur_data.iloc[i]["text_b"]

                    fake_targets = []
                    length = len(cur_data)
                    real_target = target
                    assert real_target is not None

                    while True:
                        randn = random.randint(0, length - 1)
                        sample_target = cur_data.iloc[randn]["target"]

                        text_a_samples = cur_data[cur_data["text_a"] == text_a]
                        cur_targets = text_a_samples["target"].values

                        if sample_target != real_target:
                            if sample_target not in cur_targets:
                                fake_targets.append(sample_target)
                                fake_target = sample_target
                                break
                    example = InputExample(guid=guid,
                                           text_a=text_a.replace("<blank>", "<mask>"),
                                           text_b=text_b,
                                           target=fake_target)
                    examples.append(example)
                    if (index + 1) % print_interval == 0:
                        self.logger.info(pbar(step=index))
            self.logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def create_features(self, examples, max_seq_len, max_decode_len, his_len, cached_features_file, example_type=None):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''
        pbar = ProgressBar(n_total=len(examples), desc='create features')
        print_interval = max(1, len(examples) // 10)
        if cached_features_file.exists():
            self.logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id, example in enumerate(examples):
                tokens_a = self.tokenizer(example.text_a, return_tensors='pt')["input_ids"]
                # print(example.target)
                labels = self.tokenizer(str(example.target), return_tensors='pt')["input_ids"]
                tokens_b = None

                if example.text_b is not None:
                    text_b = example.text_b
                    tokens_b = self.tokenizer(text_b, return_tensors='pt')["input_ids"]
                    tokens_a, tokens_b = self.truncate_seq_pair(tokens_a, tokens_b, max_length=max_seq_len)
                    input_ids = tokens_a + tokens_b
                else:
                    if len(tokens_a) > max_seq_len:
                        tokens_a = tokens_a[:max_seq_len]
                    input_ids = tokens_a
                    input_ids = input_ids.squeeze().tolist()

                if len(input_ids) > max_seq_len:
                    print(f"length of inputs larger than max_lenght: {input_ids}")

                if tokens_b is not None:
                    segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)
                else:
                    segment_ids = [0] * len(input_ids)

                input_mask = [1] * len(input_ids)
                input_len = len(input_ids)

                if input_len < max_seq_len:
                    input_ids += [self.tokenizer.pad_token_id] * (max_seq_len - len(input_ids))
                    input_mask += [0] * (max_seq_len - input_len)
                    segment_ids += [0] * (max_seq_len - input_len)

                if len(segment_ids) != max_seq_len:
                    print(f"max_len:{max_seq_len} input_len:{input_len} "
                          f"token_a_len:{len(tokens_a)} tokens_b_len:{len(tokens_b)}")
                    print(tokens_a)
                    print(tokens_b)
                    print(input_ids)

                if len(input_ids) != max_seq_len:
                    print(f"max_len:{max_seq_len} input_len:{input_len} "
                          f"token_a_len:{len(tokens_a)} tokens_b_len:{len(tokens_b)}")
                    print(tokens_a)
                    print(tokens_b)
                    print(input_ids)

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len

                labels = labels.squeeze().tolist()
                labels_len = len(labels)
                labels_mask = [1] * labels_len

                if len(labels) >= max_decode_len:
                    labels = labels[:max_decode_len]
                    labels_mask = labels_mask[:max_decode_len]
                    labels_len = max_decode_len
                else:
                    labels += [self.tokenizer.pad_token_id] * (max_decode_len - labels_len)
                    labels_mask += [0] * (max_decode_len - labels_len)

                assert len(labels) == max_decode_len
                assert len(labels_mask) == max_decode_len

                if example_type is None:
                    cls_label = 0
                else:
                    assert example_type == "fake"
                    cls_label = 1

                if ex_id < 2:
                    self.logger.info("*** Example ***")
                    self.logger.info(f"guid: {example.guid}" % ())
                    self.logger.info(f"tokens: {example.text_a}")
                    self.logger.info(f"relation: {example.text_b}")
                    self.logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    self.logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
                    self.logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
                    self.logger.info(f"label_ids: {' '.join(([str(x) for x in labels]))}")
                    self.logger.info(f"labels: {example.target}")
                    self.logger.info(f"cls_label: {cls_label}")

                feature = InputFeature(input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       label_ids=labels,
                                       input_len=input_len,
                                       labels_mask=labels_mask,
                                       labels_len=labels_len,
                                       cls_label=cls_label)
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
            self.logger.info("sorted data by th length of input")
            features = sorted(features, key=lambda x: x.input_len, reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        all_labels_mask = torch.tensor([f.labels_mask for f in features], dtype=torch.long)
        all_labels_len = torch.tensor([f.labels_len for f in features], dtype=torch.long)
        all_cls_label = torch.tensor([f.cls_label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids,
                                all_input_mask,
                                all_segment_ids,
                                all_label_ids,
                                all_input_lens,
                                all_labels_mask,
                                all_labels_len,
                                all_cls_label)
        return dataset

