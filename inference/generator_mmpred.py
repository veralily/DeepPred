import torch
from .tools import model_device, ProgressBar


class Generator(object):
    def __init__(self, model, logger, n_gpu):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu=n_gpu, model=self.model)

    def predictor(self, data):
        pbar = ProgressBar(n_total=len(data), desc='Testing')
        print_interval = len(data) // 10
        all_labels = []
        all_ids = []
        for step, batch in enumerate(data):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                all_inputs_ids = batch[0]
                input_ids = all_inputs_ids[:, :, :-1]
                target_ids = all_inputs_ids[:, :, 1:]
                target_list = [target_ids[:, i, :] for i in range(target_ids.size(1))]
                logits_list = self.model(input_ids)
                event_ids_list = [torch.argmax(logits_list[i].softmax(-1), dim=-1) for i in range(self.model.num_tasks)]
                all_ids.append(torch.cat([e.unsqueeze(0) for e in event_ids_list], dim=0))
                # list of [num_tasks, B, ln]
                all_labels.append(torch.cat([e.unsqueeze(0) for e in target_list], dim=0))

            if (step + 1) % max(1, print_interval) == 0:
                show_info = pbar(step=step, info={})
                self.logger.info(show_info)

        ids = torch.cat(all_ids, dim=1)
        targets = torch.cat(all_labels, dim=1)
        return ids, targets

    # def generate(self, data, max_length, num_beams, num_return_sequences=1, length_penalty=None,
    #              early_stopping=True, do_sample=False, save_file="bart_test.csv"):
    #     pbar = ProgressBar(n_total=len(data), desc='Testing')
    #     print_interval = len(data) // 10
    #     all_input_sents = []
    #     all_label_sents = []
    #     all_generated_sents = []
    #     all_encoder_hidden_states = []
    #     for step, batch in enumerate(data):
    #         batch = tuple(t.to(self.device) for t in batch)
    #         with torch.no_grad():
    #             all_inputs_ids = batch[0]
    #             input_ids = all_inputs_ids[:, :, :-1]
    #             target_ids = all_inputs_ids[:, :, 1:]
    #             target_list = [target_ids[:, i, :] for i in range(target_ids.size(1))]
    #             logits_list = self.model(input_ids)
    #
    #             encoder_hidden_states = outputs.encoder_last_hidden_state  # [bsz, ln, E]
    #             output_ids = self.model.generate(input_ids,
    #                                              max_length=max_length,
    #                                              num_return_sequences=num_return_sequences,
    #                                              num_beams=num_beams,
    #                                              length_penalty=length_penalty,
    #                                              early_stopping=early_stopping,
    #                                              do_sample=do_sample)
    #
    #         generated_sent = self.tok.batch_decode(output_ids, skip_special_tokens=True)
    #         batch_label_strs = self.tok.batch_decode(label_ids, skip_special_tokens=True)
    #         batch_input_strs = self.tok.batch_decode(input_ids, skip_special_tokens=True)
    #         all_input_sents += batch_input_strs
    #         all_label_sents += batch_label_strs
    #         all_generated_sents += generated_sent
    #         all_encoder_hidden_states.append(encoder_hidden_states)
    #         if (step + 1) % max(1, print_interval) == 0:
    #             show_info = pbar(step=step)
    #             self.logger.info(show_info)
    #
    #     if 'cuda' in str(self.device):
    #         torch.cuda.empty_cache()
    #     return all_label_sents, all_generated_sents, all_encoder_hidden_states
