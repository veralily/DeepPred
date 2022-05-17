from tkinter import N
import torch
import time
from argparse import ArgumentParser
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from .inference.sampler import GreedySampler, TopKSampler, BeamSampler
from .inference.sampler_config import CFG
from .Data.interactive_data import create_dense_feature
from .inference.generator_CoEP_interactive import Generator

from .Models.bart_for_generation import BartWithClassification
from .Models.CoEP import CoEP


def set_sampler(cfg, sampling_algorithm):
    if "beam" in sampling_algorithm:
        cfg.set_numbeams(int(sampling_algorithm.split("-")[1]))
        sampler = BeamSampler(cfg)
    elif "topk" in sampling_algorithm:
        cfg.set_topK(int(sampling_algorithm.split("-")[1]))
        sampler = TopKSampler(cfg)
    else:
        sampler = GreedySampler(cfg)

    return sampler


class CoEPInteractive():
    def __init__(self, logger) -> None:
        super(CoEPInteractive, self).__init__()
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

    def preprocess_test_data(self, sentences, max_seq_len, history_length, tokenizer, batch_size=1):
        from Data.utils import collate_fn_test
        from Data.interactive_data import create_test_dataset

        test_dataset = create_test_dataset(sentences, tokenizer, max_seq_len, history_length)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size,
                                 collate_fn=collate_fn_test)
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

    def run_generate(self, model, context, event, tok, max_seq_len=50, max_decode_len=20,history_length=100, sampling_algorithm="beam-4",
                    length_penalty=1.0):
        generator = Generator(model=model, tok=tok, logger=self.logger, n_gpu=None)
        cfg = CFG(max_decode_len, tok)
        sampler = set_sampler(cfg, sampling_algorithm)
        data = create_dense_feature({'context': context, 'event': event}, tokenizer=tok,
                                        max_seq_len=max_seq_len,
                                        his_len=history_length)

        result_data, input_sent, seqs = generator.interactive_generate_example(
                data=data,
                sampler=sampler,
                max_length=max_decode_len,
                repetition_penalty=None,
                length_penalty=length_penalty,
                no_repeat_ngram_size=None,
                num_return_sequences=int(sampling_algorithm.split('-')[-1]))

        return seqs

    def run_explanation(self, model, kg_model, context, event, tok, max_seq_len=50, max_decode_len=20,history_length=100, sampling_algorithm="beam-4",
                    length_penalty=1.0):
        generator = Generator(model=model, tok=tok, logger=self.logger, n_gpu=None)
        cfg = CFG(max_decode_len, tok)
        sampler = set_sampler(cfg, sampling_algorithm)
        data = create_dense_feature({'context': context, 'event': event}, tokenizer=tok,
                                        max_seq_len=max_seq_len,
                                        his_len=history_length)                  

        input_sent, seqs = generator.interactive_generate_explanation(
                data=data,
                KG_model=kg_model,
                sampler=sampler,
                max_length=max_decode_len,
                repetition_penalty=None,
                length_penalty=length_penalty,
                no_repeat_ngram_size=None,
                num_return_sequences=int(sampling_algorithm.split('-')[-1]))

        return seqs


# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--arch", default='bart-base', type=str)
#     parser.add_argument('--logger_info', default='', type=str)
#     parser.add_argument('--filename', default='dataset/event_story/event_story_test_refs.csv', type=str)
#     parser.add_argument('--output_file', default='generate', type=str)
#     parser.add_argument('--model_path', default='output/checkpoints/bart-base_event_story_ECoGM_2e-5/'
#                                                 'epoch_8_bart-base_model.bin', type=str)

#     parser.add_argument('--sampling_algorithm', default='beam-4', type=str)
#     parser.add_argument('--length_penalty', default=1.0, type=float)
#     parser.add_argument('--repetition_penalty', default=2.0, type=float)
#     parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')

#     parser.add_argument("--history_length", type=int, default=100)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--max_seq_len", default=50, type=int)
#     parser.add_argument("--max_decode_len", default=20, type=int)

#     args = parser.parse_args()

#     init_logger(
#         log_file=f"{config['log_dir']}/{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}-{args.logger_info}.log")
#     logger.info("Generation parameters %s", args)

#     kg_model = load_atomic_KG_model(args.n_gpu)
#     GM = BartForConditionalGeneration.from_pretrained(str(config[f'{args.arch}_model_dir']))

#     ecogm = CoEP(bart_config=kg_model.config, pretrained_im=kg_model.model, pretrained_gm=GM, num_labels=2)
#     if args.model_path is None:
#         ecogm.load_state_dict(torch.load(ft_config['ECoGM'], map_location=f"cuda:{args.n_gpu}")['model'])
#     else:
#         ecogm.load_state_dict(torch.load(args.model_path, map_location=f"cuda:{args.n_gpu}")['model'])

#     tok = BartTokenizer.from_pretrained(str(config[f'{args.arch}_model_dir']))
#     cfg = CFG(args.max_decode_len, tok)
#     sampler = set_sampler(cfg, args.sampling_algorithm)
#     generator = Generator(model=ecogm, tok=tok, logger=logger, n_gpu=args.n_gpu)
#     num_return_sequences = 1

#     if args.filename is None:
#         while input("stop generating? ") != "yes":
#             context = input("context: ")
#             event = input("event: ")

#             data = create_dense_feature({'context': context, 'event': event}, tokenizer=tok,
#                                         max_seq_len=args.max_seq_len,
#                                         his_len=args.history_length)
#             # ----------- predicting ----------
#             logger.info('interactive generating....')

#             result_data, input_sent, seqs = generator.interactive_generate_example(
#                 data=data,
#                 sampler=sampler,
#                 max_length=args.max_decode_len,
#                 repetition_penalty=None,
#                 length_penalty=args.length_penalty,
#                 no_repeat_ngram_size=None,
#                 num_return_sequences=int(args.sampling_algorithm.split('-')[-1]))

#             logger.info(f"input: {input_sent}, generated: {seqs}")
#     else:
#         import pandas as pd
#         stories = pd.read_csv(args.filename)
#         sentences = [[story.split('\t')[1]] for story in stories['context'].values]

#         test_dataloader = preprocess_test_data(sentences, args=args, tokenizer=tok)

#         for i in range(4):
#             all_generated_sents = generator.generate_example(data=test_dataloader,
#                                                              sampler=sampler,
#                                                              max_length=args.max_decode_len,
#                                                              repetition_penalty=args.repetition_penalty,
#                                                              length_penalty=args.length_penalty,
#                                                              num_return_sequences=num_return_sequences,
#                                                              no_repeat_ngram_size=None)
#             print(f"type all_generated_sents: {type(all_generated_sents)}\nsample: {all_generated_sents[0:10]}")
#             assert len(sentences) == len(all_generated_sents)

#             for i in range(len(sentences)):
#                 sentences[i].append(str(all_generated_sents[i]))
#             # sentences = [list(s).extend([str(g)]) for s, g in zip(sentences, all_generated_sents)]
#             print(f"type sentences: {type(sentences)}\n type sentence:{type(sentences[0])}\nsample: {sentences[0:10]}")
#             test_dataloader = preprocess_test_data(sentences=sentences, args=args, tokenizer=tok)

#         df = pd.DataFrame(columns=["generated"])
#         df["generated"] = [' '.join(s) for s in sentences]
#         df.to_csv(f'{args.output_file}.csv')


# if __name__ == '__main__':
#     main()
