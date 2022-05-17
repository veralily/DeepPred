import torch
import pandas as pd

from ..callback.progressbar import ProgressBar
from ..tools import model_device
from ..tools import seed_everything
from ..tools import AverageMeter

from .losses import huber_loss, sequence_loss


class Trainer(object):
    """
    The optimizer and scheduler are dictionaries of several itmes with different params
    """

    def __init__(self, args, model, logger, optimizer, scheduler, early_stopping,
                 event_epoch_metrics, time_epoch_metrics, event_batch_metrics, time_batch_metrics,
                 training_monitor=None, model_checkpoint=None, resume_path=None, ):
        self.args = args
        self.model = model
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.input_length = args.input_length
        self.output_length = args.output_length

        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.event_epoch_metrics = event_epoch_metrics
        self.time_epoch_metrics = time_epoch_metrics
        self.event_batch_metrics = event_batch_metrics
        self.time_batch_metrics = time_batch_metrics
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

        self.outputs_events = []
        self.outputs_times = []
        self.target_events = []
        self.target_times = []
        self.result = {}
        self.info = {}
        self.epoch_loss_agg = AverageMeter()
        self.epoch_event_loss_agg = AverageMeter()
        self.epoch_time_loss_agg = AverageMeter()
        self.epoch_gen_loss_agg = AverageMeter()
        self.epoch_disc_loss_agg = AverageMeter()

    def epoch_reset(self):
        self.outputs_events = []
        self.outputs_times = []
        self.target_events = []
        self.target_times = []
        self.result = {}
        self.epoch_loss_agg = AverageMeter()
        self.epoch_event_loss_agg = AverageMeter()
        self.epoch_time_loss_agg = AverageMeter()
        self.epoch_gen_loss_agg = AverageMeter()
        self.epoch_disc_loss_agg = AverageMeter()
        if self.event_epoch_metrics is not None:
            for metric in self.event_epoch_metrics:
                metric.reset()

        if self.time_epoch_metrics is not None:
            for metric in self.time_epoch_metrics:
                metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.event_batch_metrics:
            metric.reset()

        for metric in self.time_batch_metrics:
            metric.reset()

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save.state_dict(),
                 'epoch': epoch,
                 'best': best}
        return state

    def freeze_grad(self, modules):
        for module in modules:
            for name, params in module.named_parameters():
                params.requires_grad = False

    def unfreeze_grad(self, modules):
        for module in modules:
            for name, params in module.named_parameters():
                params.requires_grad = True

    def get_gradients(self, module):
        for name, parames in module.named_parameters():
            print(f"name:{name}, params: {parames}")

    def run_batch(self, batch, step, sample_batch, mode):
        self.batch_reset()
        batch = tuple(t.to(self.device) for t in batch)
        sample_batch = tuple(t.to(self.device) for t in sample_batch)
        all_inputs_event, all_inputs_time = batch  # [B, ln]
        _, sample_time = sample_batch

        inputs_event = all_inputs_event[:, :self.input_length]
        labels_event = all_inputs_event[:, self.input_length:]
        inputs_time = all_inputs_time[:, :self.input_length]
        labels_time = all_inputs_time[:, self.input_length:]

        logits_event, logits_time, gumbel_softmax_time = self.model(inputs_event, inputs_time, self.output_length)
        event_loss = sequence_loss(logits_event, labels_event)
        time_loss = huber_loss(gumbel_softmax_time, labels_time)
        disc_fake = self.model.discriminator(torch.cat([inputs_time, logits_time], dim=-1))
        gen_time_loss = -disc_fake.mean()
        gen_loss = self.alpha * event_loss + gen_time_loss + self.gamma * time_loss
        disc_real = self.model.discriminator(sample_time)
        disc_loss = -(disc_real.mean() - disc_fake.mean())

        if mode == "train-event":
            # train event encoder and decoder
            if self.args.gradient_accumulation_steps > 1:
                event_loss = event_loss / self.args.gradient_accumulation_steps
            event_loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer['event'].step()
                self.scheduler['event'].step()
                self.optimizer['event'].zero_grad()

        elif mode == "train-time":
            # train time encoder and decoder
            if self.args.gradient_accumulation_steps > 1:
                time_loss = time_loss / self.args.gradient_accumulation_steps
            time_loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer['time'].step()
                self.scheduler['time'].step()
                self.optimizer['time'].zero_grad()

        elif mode == "train-gen":
            # train all generative modules
            if self.args.gradient_accumulation_steps > 1:
                gen_loss = gen_loss / self.args.gradient_accumulation_steps
            gen_loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer['gen'].step()
                self.scheduler['gen'].step()
                self.optimizer['gen'].zero_grad()
            self.global_step += 1

        elif mode == "train-disc":

            if self.args.gradient_accumulation_steps > 1:
                disc_loss = disc_loss / self.args.gradient_accumulation_steps
            disc_loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer['disc'].step()
                self.scheduler['disc'].step()
                self.optimizer['disc'].zero_grad()
            self.global_step += 1

        else:  # mode == valid
            event_loss = event_loss
            time_loss = time_loss
            gen_loss = gen_loss
            disc_loss = disc_loss

        self.info[f'event-loss'] = event_loss.item()
        self.info[f'time-loss'] = time_loss.item()
        self.info[f'gen-loss'] = gen_loss.item()
        self.info[f'disc-loss'] = disc_loss.item()
        self.epoch_loss_agg.update((event_loss+time_loss).item(), n=1)
        self.epoch_event_loss_agg.update(event_loss.item(), n=1)
        self.epoch_time_loss_agg.update(time_loss.item(), n=1)
        self.epoch_gen_loss_agg.update(gen_loss.item(), n=1)
        self.epoch_disc_loss_agg.update(disc_loss.item(), n=1)

        # assert next_tokens has shape [batch_size, length]
        event_ids = torch.argmax(logits_event.softmax(-1), dim=-1)

        return {"event_ids": event_ids,
                "logits_event": logits_event,
                "target_event": labels_event,
                "logits_time": logits_time,
                "target_time": labels_time}

    def model_fit(self, data, samples_data, print_interval, pbar, gap, is_training=True):
        for step, (batch, sample_batch) in enumerate(zip(data, samples_data)):
            if not is_training:
                batch_outputs = self.run_batch(batch, step, sample_batch, mode="valid")
            else:
                if step % gap == 0:
                    batch_outputs = self.run_batch(batch, step, sample_batch, mode="train-disc")
                else:
                    batch_outputs = self.run_batch(batch, step, sample_batch, mode="train-event")
                    batch_outputs = self.run_batch(batch, step, sample_batch, mode="train-time")
                    batch_outputs = self.run_batch(batch, step, sample_batch, mode="train-gen")

            event_ids = batch_outputs["event_ids"]
            logits_event = batch_outputs["logits_event"]
            target_event = batch_outputs["target_event"]
            logits_time = batch_outputs["logits_time"]
            target_time = batch_outputs["target_time"]

            if self.event_batch_metrics:
                for metric in self.event_batch_metrics:
                    metric(logits=logits_event.contiguous().view(-1, logits_event.shape[-1]),
                           target=target_event.contiguous().view(-1))
                    self.info[metric.name()] = metric.value()

            if self.time_batch_metrics:
                for metric in self.time_batch_metrics:
                    metric(logits_time.contiguous().view(-1),
                           target_time.contiguous().view(-1))
                    self.info[metric.name()] = metric.value()

            if step % print_interval == 0:
                show_info = pbar(step=step, info=self.info)
                self.logger.info(show_info)

            self.outputs_events.append(event_ids)
            self.target_events.append(target_event)
            self.outputs_times.append(logits_time)
            self.target_times.append(target_time)

    def run_epoch(self, data, samples_data, gap, filename=None, is_training=True):
        length_data = len(data)
        print_interval = max(length_data // 10, 1)
        self.epoch_reset()
        if is_training:
            self.model.train()
            pbar = ProgressBar(n_total=length_data, desc='Training')
            self.model_fit(data, samples_data, print_interval=print_interval, gap=gap, pbar=pbar)
            self.logger.info("\n------------- train result --------------")
            self.result['loss'] = self.epoch_loss_agg.avg
            self.result['event-loss'] = self.epoch_event_loss_agg.avg
            self.result['time-loss'] = self.epoch_time_loss_agg.avg
            self.result['gen-loss'] = self.epoch_gen_loss_agg.avg
            self.result['disc-loss'] = self.epoch_disc_loss_agg.avg

        else:
            self.model.eval()
            pbar = ProgressBar(n_total=length_data, desc='Evaluating')
            with torch.no_grad():
                self.model_fit(data, samples_data, gap=gap, print_interval=print_interval, pbar=pbar, is_training=False)

                print("------------- valid result --------------")
                self.result['valid_loss'] = self.epoch_loss_agg.avg
                self.result['valid_event-loss'] = self.epoch_event_loss_agg.avg
                self.result['valid_time-loss'] = self.epoch_time_loss_agg.avg
                self.result['valid_gen-loss'] = self.epoch_gen_loss_agg.avg
                self.result['valid_disc-loss'] = self.epoch_disc_loss_agg.avg

        if self.event_epoch_metrics:
            outputs_events = torch.cat(self.outputs_events, dim=0)
            target_events = torch.cat(self.outputs_times, dim=0)
            for metric in self.event_epoch_metrics:
                metric(outputs_events,
                       target_events)
                value = metric.value()
                self.result[f'valid_{metric.name()}'] = value

        if self.time_epoch_metrics:
            outputs_times = torch.cat(self.outputs_times, dim=0)
            target_times = torch.cat(self.target_times, dim=0)
            for metric in self.time_epoch_metrics:
                metric(outputs_times, target_times)
                value = metric.value()
                self.result[f'valid_{metric.name()}'] = value

        if filename is not None:
            self.logger.info(f"Save valid cases to {filename}!")
            self.save_case(self.outputs_events, self.target_events, filename)

        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result

    def train(self, train_data, train_samples, valid_data, valid_samples, gap):
        # ***************************************************************
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 a
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")

            train_log = self.run_epoch(train_data, train_samples, gap=gap, is_training=True)

            valid_log = self.run_epoch(valid_data, valid_samples, gap=gap, is_training=False)
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
