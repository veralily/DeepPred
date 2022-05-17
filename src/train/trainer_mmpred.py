import torch
import pandas as pd

from ..callback.progressbar import ProgressBar
from ..tools import model_device
from ..tools import seed_everything
from ..tools import AverageMeter

from .losses import sequence_loss


class Trainer(object):
    """
    The optimizer and scheduler are dictionaries of several itmes with different params
    """

    def __init__(self, args, model, logger, optimizer, scheduler, early_stopping,
                 epoch_metrics_list, batch_metrics_list,
                 training_monitor=None, model_checkpoint=None, resume_path=None):
        self.args = args
        self.model = model
        self.num_tasks = model.num_tasks

        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.epoch_metrics_list = epoch_metrics_list
        self.batch_metrics_list = batch_metrics_list
        self.training_monitor = training_monitor
        self.model_checkpoint = model_checkpoint

        self.start_epoch = 1
        self.global_step = 0
        self.model, self.device = model_device(n_gpu=args.n_gpu, model=self.model)

        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(f"{resume_path}")
            best = resume_dict['best']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

        self.outputs_list = []
        self.target_list = []
        self.result = {}
        self.info = {}
        self.epoch_loss_agg = AverageMeter()

    def epoch_reset(self):
        self.outputs_list = []
        self.target_list = []
        self.result = {}
        self.epoch_loss_agg = AverageMeter()
        if self.epoch_metrics_list is not None:
            for epoch_metrics in self.epoch_metrics_list:
                for metric in epoch_metrics:
                    metric.reset()

    def batch_reset(self):
        self.info = {}
        for batch_metrics in self.batch_metrics_list:
            for metric in batch_metrics:
                metric.reset()

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save.state_dict(),
                 'epoch': epoch,
                 'best': best}
        return state

    def run_batch(self, batch, step, mode):
        self.batch_reset()
        batch = tuple(t.to(self.device) for t in batch)
        all_inputs_ids = batch[0]  # [B, num_tasks, len]
        input_ids = all_inputs_ids[:, :, :-1]
        target_ids = all_inputs_ids[:, :, 1:]
        target_list = [target_ids[:, i, :] for i in range(target_ids.size(1))]

        logits_list = self.model(input_ids)
        # list of [batch_size, ln, vocab_size]
        loss_list = [sequence_loss(logits_list[i], target_list[i]) for i in range(self.num_tasks)]
        loss = sum(loss_list)

        if mode == "train":
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            self.global_step += 1

        else:  # mode == valid
            loss = loss

        self.info[f'loss'] = loss.item()
        self.epoch_loss_agg.update(loss.item(), n=1)

        # assert next_tokens has shape [batch_size, length]
        event_ids_list = [torch.argmax(logits_list[i].softmax(-1), dim=-1) for i in range(self.num_tasks)]

        return {"event_ids_list": event_ids_list,
                "logits_list": logits_list,
                "target_list": target_list}

    def model_fit(self, data, print_interval, pbar, is_training=True):
        for step, batch in enumerate(data):
            if not is_training:
                batch_outputs = self.run_batch(batch, step=step, mode="valid")
            else:
                batch_outputs = self.run_batch(batch, step=step, mode="train")

            event_ids_list = batch_outputs["event_ids_list"]
            logits_list = batch_outputs["logits_list"]
            target_list = batch_outputs["target_list"]

            if self.batch_metrics_list:
                for i in range(self.num_tasks):
                    for metric in self.batch_metrics_list[i]:
                        metric(logits_list[i].contiguous().view(-1, logits_list[i].shape[-1]),
                               target_list[i].contiguous().view(-1))
                        self.info[f"{metric.name()}-task-{i}"] = metric.value()

            if step % print_interval == 0:
                show_info = pbar(step=step, info=self.info)
                self.logger.info(show_info)

            self.outputs_list.append(torch.cat([e.unsqueeze(0) for e in event_ids_list], dim=0))
            # list of [num_tasks, B, ln]
            self.target_list.append(torch.cat([e.unsqueeze(0) for e in target_list], dim=0))
            # list of [num_tasks, B, ln]

    def run_epoch(self, data, filename=None, is_training=True):
        length_data = len(data)
        print_interval = max(length_data // 10, 1)
        self.epoch_reset()
        if is_training:
            self.model.train()
            pbar = ProgressBar(n_total=length_data, desc='Training')
            self.model_fit(data, print_interval=print_interval, pbar=pbar)
            self.logger.info("------------- train result --------------")
            self.result['loss'] = self.epoch_loss_agg.avg

        else:
            self.model.eval()
            pbar = ProgressBar(n_total=length_data, desc='Evaluating')
            with torch.no_grad():
                self.model_fit(data, print_interval=print_interval, pbar=pbar, is_training=False)
                self.logger.info(f"------------- valid result --------------")
                self.result['valid_loss'] = self.epoch_loss_agg.avg

        logits = torch.cat(self.outputs_list, dim=1)
        target = torch.cat(self.target_list, dim=1)

        if self.epoch_metrics_list:
            # [num_tasks, num_instances]
            for i in range(self.num_tasks):
                for metric in self.epoch_metrics_list[i]:
                    metric(logits[i, :, -1], target[i, :, -1])
                    key_name = f"{metric.name()}-task-{i}" if is_training else f"valid_{metric.name()}-task-{i}"
                    self.result[key_name] = metric.value()

        if filename is not None:
            self.logger.info(f"Save valid cases to {filename}!")
            self.save_case(self.outputs_list, self.target_list, filename)

        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self, train_data, valid_data):
        # ***************************************************************
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 a
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")

            train_log = self.run_epoch(train_data, is_training=True)
            # valid_result_dir = f'{config["result_dir"]}/{self.args.loginfo}'
            # import os
            # if os.path.exists(valid_result_dir):
            #     pass
            # else:
            #     os.mkdir(valid_result_dir)

            valid_log = self.run_epoch(valid_data, is_training=False)
            logs = dict(train_log, **valid_log)

            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(show_info)

            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                self.logger.info(f"save model_checkpoints")
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break

    def save_case(self, pred, label, file):
        assert len(pred) == len(label)
        df_gen = pd.DataFrame(columns=['generated', 'reference'])
        df_gen['generated'] = pred
        df_gen['reference'] = label
        df_gen.to_csv(f"{file}.csv")
