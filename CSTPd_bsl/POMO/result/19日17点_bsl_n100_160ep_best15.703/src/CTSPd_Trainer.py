
import csv
import json
import os
import time
import torch
from logging import getLogger

from CSTPd_bsl.POMO.CTSPd_Env import CTSPdEnv as Env
from CSTPd_bsl.POMO.CTSPd_Model import CTSPdModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but unavailable. Falling back to CPU.")
                USE_CUDA = False
            elif cuda_device_num >= torch.cuda.device_count():
                raise RuntimeError(
                    "CUDA device {} requested, but only {} CUDA device(s) are visible.".format(
                        cuda_device_num,
                        torch.cuda.device_count(),
                    )
                )

        if USE_CUDA:
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')
        torch.set_default_dtype(torch.float32)
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device(device)
        elif device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

        # Main Components
        self.model = Model(**self.model_params).to(device)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.early_stopping_params = self.trainer_params.get('early_stopping', {})
        self.early_stopping_state = {
            'best_value': None,
            'best_epoch': 0,
            'wait': 0,
            'should_stop': False,
        }
        self.training_elapsed_time_sec = float(self.trainer_params.get('training_elapsed_time_sec', 0.0))
        self._metrics_header_written = False

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            if 'early_stopping_state' in checkpoint:
                self.early_stopping_state.update(checkpoint['early_stopping_state'])
            self.training_elapsed_time_sec = float(
                checkpoint.get('training_elapsed_time_sec', self.training_elapsed_time_sec)
            )
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        self.log_image_available = None

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        self.training_start_time = time.time() - self.training_elapsed_time_sec
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            epoch_start_time = time.time()
            learning_rate = self.optimizer.param_groups[0]['lr']

            self._set_model_training_epoch(epoch)

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            early_stop = self._update_early_stopping(epoch, train_score, train_loss)
            epoch_time_sec = time.time() - epoch_start_time
            total_training_time_sec = time.time() - self.training_start_time
            avg_epoch_time_sec = total_training_time_sec / epoch
            self._write_training_progress(
                epoch,
                train_score,
                train_loss,
                early_stop,
                epoch_time_sec,
                total_training_time_sec,
                avg_epoch_time_sec,
                learning_rate,
            )

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            self.logger.info(
                "Epoch {:3d}: epoch_time: {:.2f} sec, total_time: {:.2f} sec, avg_epoch_time: {:.2f} sec".format(
                    epoch,
                    epoch_time_sec,
                    total_training_time_sec,
                    avg_epoch_time_sec,
                )
            )

            all_done = early_stop or (epoch == self.trainer_params['epochs'])
            model_save_interval = max(1, int(self.trainer_params['logging'].get('model_save_interval', 1)))
            img_save_interval = int(self.trainer_params['logging'].get('img_save_interval', 0) or 0)

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                self._save_log_images(image_prefix)

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                self._save_checkpoint(epoch)

            if img_save_interval and (all_done or (epoch % img_save_interval) == 0):
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                self._save_log_images(image_prefix)

            if all_done:
                if early_stop:
                    self.logger.info(" *** Early stopping triggered *** ")
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                break

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            if prob is None:
                raise RuntimeError("Model returned no probability while training.")
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        if reward is None:
            raise RuntimeError("Environment finished without producing a reward.")

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return float(score_mean.item()), float(loss_mean.item())

    def _save_log_images(self, image_prefix):
        if not self._can_save_log_images():
            return

        util_save_log_image_with_label(
            image_prefix,
            self.trainer_params['logging']['log_image_params_1'],
            self.result_log,
            labels=['train_score'],
        )
        util_save_log_image_with_label(
            image_prefix,
            self.trainer_params['logging']['log_image_params_2'],
            self.result_log,
            labels=['train_loss'],
        )

    def _can_save_log_images(self):
        if self.log_image_available is not None:
            return self.log_image_available

        self.log_image_available = util_can_save_log_images(self.logger)
        return self.log_image_available

    def _save_checkpoint(self, epoch):
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'result_log': self.result_log.get_raw_data(),
            'early_stopping_state': self.early_stopping_state,
            'training_elapsed_time_sec': time.time() - self.training_start_time,
        }
        checkpoint_path = '{}/checkpoint-{}.pt'.format(self.result_folder, epoch)
        torch.save(checkpoint_dict, checkpoint_path)
        torch.save(checkpoint_dict, '{}/checkpoint-latest.pt'.format(self.result_folder))
        self.logger.info("Saved checkpoint: %s", checkpoint_path)

    def _save_best_checkpoint(self, epoch):
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'result_log': self.result_log.get_raw_data(),
            'early_stopping_state': self.early_stopping_state,
            'training_elapsed_time_sec': time.time() - self.training_start_time,
        }
        checkpoint_path = '{}/checkpoint-best.pt'.format(self.result_folder)
        torch.save(checkpoint_dict, checkpoint_path)
        self.logger.info("Saved best checkpoint: %s", checkpoint_path)

    def _set_model_training_epoch(self, epoch):
        setter = getattr(self.model, 'set_training_epoch', None)
        if callable(setter):
            setter(epoch, self.trainer_params['epochs'])

    def _update_early_stopping(self, epoch, train_score, train_loss):
        params = self.early_stopping_params
        if not params.get('enable', False):
            return False

        monitor = params.get('monitor', 'train_score')
        mode = params.get('mode', 'min')
        min_delta = float(params.get('min_delta', 1e-4))
        patience = int(params.get('patience', 50))
        warmup_epochs = int(params.get('warmup_epochs', 0))
        checkpoint_best = params.get('checkpoint_best', True)

        metrics = {
            'train_score': train_score,
            'train_loss': train_loss,
        }
        if monitor not in metrics:
            raise ValueError("Unsupported early stopping monitor: {}".format(monitor))

        current = float(metrics[monitor])
        best = self.early_stopping_state['best_value']
        if best is None:
            improved = True
        elif mode == 'min':
            improved = current < float(best) - min_delta
        elif mode == 'max':
            improved = current > float(best) + min_delta
        else:
            raise ValueError("Unsupported early stopping mode: {}".format(mode))

        if improved:
            self.early_stopping_state['best_value'] = current
            self.early_stopping_state['best_epoch'] = epoch
            self.early_stopping_state['wait'] = 0
            if checkpoint_best:
                self._save_best_checkpoint(epoch)
            return False

        if epoch <= warmup_epochs:
            self.early_stopping_state['wait'] = 0
            return False

        self.early_stopping_state['wait'] += 1
        wait = self.early_stopping_state['wait']
        self.logger.info(
            "Early stopping monitor[%s]: current=%.6f, best=%.6f at epoch %d, wait=%d/%d",
            monitor,
            current,
            float(self.early_stopping_state['best_value']),
            self.early_stopping_state['best_epoch'],
            wait,
            patience,
        )
        should_stop = wait >= patience
        self.early_stopping_state['should_stop'] = should_stop
        return should_stop

    def _write_training_progress(
            self,
            epoch,
            train_score,
            train_loss,
            early_stop,
            epoch_time_sec,
            total_training_time_sec,
            avg_epoch_time_sec,
            learning_rate):
        metrics_path = os.path.join(self.result_folder, 'training_metrics.csv')
        write_header = (not self._metrics_header_written) and (not os.path.exists(metrics_path))
        with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'epoch',
                    'train_score',
                    'train_loss',
                    'best_epoch',
                    'best_value',
                    'early_stop_wait',
                    'learning_rate',
                    'epoch_time_sec',
                    'total_training_time_sec',
                    'avg_epoch_time_sec',
                ])
            writer.writerow([
                epoch,
                train_score,
                train_loss,
                self.early_stopping_state['best_epoch'],
                self.early_stopping_state['best_value'],
                self.early_stopping_state['wait'],
                learning_rate,
                epoch_time_sec,
                total_training_time_sec,
                avg_epoch_time_sec,
            ])
        self._metrics_header_written = True

        progress = {
            'epoch': epoch,
            'total_epochs': self.trainer_params['epochs'],
            'train_score': train_score,
            'train_loss': train_loss,
            'best_epoch': self.early_stopping_state['best_epoch'],
            'best_value': self.early_stopping_state['best_value'],
            'early_stop_wait': self.early_stopping_state['wait'],
            'early_stop': bool(early_stop),
            'epoch_time_sec': epoch_time_sec,
            'total_training_time_sec': total_training_time_sec,
            'avg_epoch_time_sec': avg_epoch_time_sec,
        }
        with open(os.path.join(self.result_folder, 'training_progress.json'), 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
