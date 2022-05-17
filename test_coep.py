import torch
import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from .src.tools import init_logger
from .configs.coep_config import config
from .src.preprocessing.preprocessor import EnglishPreProcessor

from .inference.sampler import GreedySampler, TopKSampler, BeamSampler
from .inference.sampler_config import CFG
from .Data.story_task_data import TaskData, BartProcessor
from .Data.utils import collate_fn as collate_fn
from .Models.bart_for_generation import BartWithClassification
from .Models.CoEP import CoEP
from .inference.generator_CoEP import Generator


def set_sampler(cfg, sampling_algorithm):
    if "beam" in sampling_algorithm:
        cfg.set_numbeams(int(sampling_algorithm.split("-")[1]))
        sampler = BeamSampler(cfg)
    elif "topk" in sampling_algorithm:
        # print("Still bugs in the topk sampler. Use beam or greedy instead")
        # raise NotImplementedError
        cfg.set_topK(int(sampling_algorithm.split("-")[1]))
        sampler = TopKSampler(cfg)
    else:
        sampler = GreedySampler(cfg)

    return sampler


class CoEPTest():
    def __init__(self, logger):
        super(CoEPTest, self).__init__()
        self.logger = logger

    def load_atomic_KG_wcls_model(self, config_path, model_path, device=None):
        map_location = "cpu" if device is None else f"cuda:{device}"
        IM = BartWithClassification(config_path, num_labels=2)
        states = torch.load(model_path, map_location=map_location)
        model_state_dict = states['model'] if 'model' in states else states
        IM.load_state_dict(model_state_dict)
        return IM
        

    def load_sequential_KG_model(self, config_path, device=None):
        map_location = "cpu" if device is None else f"cuda:{device}"
        model = BartForConditionalGeneration.from_pretrained(config_path)
        return model

    def load_test_data(self, file_path, cached_examples_file, cached_features_file, tokenizer, batch_size, max_seq_len,
                       max_decode_len, history_length):
        data = TaskData(raw_data_path=file_path,
                        preprocessor=EnglishPreProcessor(),
                        is_train=False)
        targets, sentences, chars = data.read_data()
        lines = list(zip(sentences, targets, chars))

        processor = BartProcessor(tokenizer=tokenizer)
        test_data = processor.get_test(lines=lines)
        test_examples = processor.create_examples(
            lines=test_data,
            data_split='inference',
            cached_examples_file=cached_examples_file)
        test_features = processor.create_features(
            examples=test_examples,
            max_seq_len=max_seq_len,
            cached_features_file=cached_features_file,
            max_decode_len=max_decode_len,
            his_len=history_length,
        )

        test_features = test_features
        test_dataset = processor.create_dataset(test_features)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size,
                                     collate_fn=collate_fn)

        self.logger.info('Loading Test Data....')
        self.logger.info("  Num examples = %d", len(test_features))

        return test_dataloader

    def load_model(self, config_path, model_path, im_path, device=None):
        num_lables = 2
        map_location = "cpu" if device is None else f"cuda:{device}"
        states = torch.load(model_path, map_location=map_location)
        
        model_state_dict = states['model'] if 'model' in states else states
        kg_bart_model = self.load_atomic_KG_wcls_model(config_path, im_path, device)
        gm_model = self.load_sequential_KG_model(config_path, device)
        model = CoEP(kg_bart_model=kg_bart_model, bart_model=gm_model, num_labels=num_lables)
        model.load_state_dict(model_state_dict)
        return model, kg_bart_model

    def run_test(self, model, kg_model, tok, test_data, max_decode_len, sampling_algorithm,
                 length_penalty=1.0,
                 device=None,
                 save_explanation_path=None,
                 save_example_path=None):

        # ----------- predicting ----------
        self.logger.info('model predicting....')
        cfg = CFG(max_decode_len, tok)
        sampler = set_sampler(cfg, sampling_algorithm)

        generator = Generator(model=model, tok=tok, logger=self.logger, n_gpu=device)

        lp = length_penalty
        # num_return_sequences = int(args.sampling_algorithm.split("-")[1])
        num_return_sequences = 1

        self.logger.info(f"Generate explanations")
        label_sents, generated_sents, _, _ = generator.generate_explanation(
            data=test_data,
            KG_model=kg_model,
            sampler=sampler,
            max_length=max_decode_len,
            repetition_penalty=None,
            length_penalty=lp,
            no_repeat_ngram_size=None,
            save_prefix=save_explanation_path,
            KG_only=True,
            num_return_sequences=num_return_sequences)

        label_sents, generated_sents, _, _ = generator.generate_example(
            data=test_data,
            sampler=sampler,
            max_length=max_decode_len,
            repetition_penalty=None,
            length_penalty=lp,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=None,
            save_prefix=save_example_path,
            add_cls=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--arch', default='bart-base', type=str)
    parser.add_argument('--data_name', default='event_story', type=str)
    parser.add_argument("--resume_path", default='output/checkpoints/bart-base_event_story_ECoGM_2e-5', type=str)
    parser.add_argument('--resume_epoch', type=int, default=8)
    parser.add_argument("--log_info", type=str, default=None)
    parser.add_argument('--relation_index', type=int, default=None)

    parser.add_argument('--sampling_algorithm', default='beam-4', type=str)
    parser.add_argument('--length_penalty', default=1.0, type=float)
    parser.add_argument("--history_length", type=int, default=100)

    parser.add_argument("--generate_explanation", action="store_true")

    parser.add_argument("--add_cls", action='store_true')

    parser.add_argument("--n_gpu", type=str, default='1', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--max_decode_len", default=20, type=int)

    args = parser.parse_args()

    logger = init_logger(
        log_file=config['log_dir'] / f'{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}-{args.log_info}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    logger.info("Evaluation parameters %s", args)

    file_path = config["data_dir"] / f'{args.data_name}/{args.data_name}_test_refs.csv'
    cached_examples_file = config['data_dir'] / f"{args.data_name}/cached_test_examples"
    cached_features_file = config[
                               'data_dir'] / f"{args.data_name}/cached_test_features_{args.max_seq_len}_{args.arch}"

    coep_model_path = f'{args.resume_path}/epoch_{args.resume_epoch}_{args.arch}_model.bin'
    tok_path = str(config[f'{args.arch}_model_dir'])
    save_explanation_path = f"test_case_{args.data_name}_{args.log_info}_{args.sampling_algorithm}_lp{args.length_penalty}.csv"
    save_example_path = f"test_case_{args.data_name}_{args.log_info}_lp{args.length_penalty}.csv"
    im_path = config['checkpoints']/ 'bart-base_v4_atomic_KG_wcls/epoch_4_bart-base_model.bin'

    tok = BartTokenizer.from_pretrained(tok_path)
    test = CoEPTest()
    test.run_test(model_path=coep_model_path,
                  im_path=im_path,
                  tok=tok,
                  test_data=test.load_test_data(file_path=file_path,
                                                cached_examples_file=cached_examples_file,
                                                cached_features_file=cached_features_file,
                                                tokenizer=tok,
                                                batch_size=args.batch_size,
                                                max_seq_len=args.max_seq_len,
                                                max_decode_len=args.max_decode_len,
                                                history_length=args.history_length,
                                                ),
                  max_decode_len=args.max_decode_len,
                  sampling_algorithm=args.sampling_algorithm,
                  length_penalty=args.length_penalty,
                  save_explanation_path=save_explanation_path,
                  save_example_path=save_example_path)


if __name__ == '__main__':
    main()
