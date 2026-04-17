import os
import subprocess
import sys
from logging import getLogger

import torch

from CSTPd_cluster.POMO.CTSPd_Env import CTSPdEnv as Env
from CSTPd_cluster.POMO.CTSPd_Model import CTSPdModel as Model

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
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(device)

        # Main Components
        self.model = Model(**self.model_params).to(device)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

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
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        self.log_image_available = None

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                self._save_log_images(image_prefix)

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if img_save_interval and (all_done or (epoch % img_save_interval) == 0):
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                self._save_log_images(image_prefix)

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

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

        path_ready, path_message = self._has_activated_conda_dll_path()
        if not path_ready:
            self.log_image_available = False
            self.logger.warning("Skipping log_image because conda DLL PATH is not activated: %s", path_message)
            return False

        code = (
            "import os, tempfile; "
            "os.environ.setdefault('MPLBACKEND', 'Agg'); "
            "import matplotlib; matplotlib.use('Agg'); "
            "import matplotlib.pyplot as plt; "
            "fd, path = tempfile.mkstemp(suffix='.png'); os.close(fd); "
            "plt.figure(); plt.plot([0, 1], [0, 1]); plt.savefig(path); "
            "plt.close('all'); os.remove(path)"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-X", "faulthandler", "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
        except Exception as exc:
            self.log_image_available = False
            self.logger.warning("Skipping log_image because Matplotlib health check failed: %s", exc)
            return False

        self.log_image_available = completed.returncode == 0
        if not self.log_image_available:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
            reason = stderr[0] if stderr else "no stderr"
            self.logger.warning(
                "Skipping log_image because Matplotlib savefig is not healthy "
                "(exit code %s): %s",
                completed.returncode,
                reason,
            )

        return self.log_image_available

    def _has_activated_conda_dll_path(self):
        env_prefix = os.environ.get('CONDA_PREFIX')
        executable_prefix = os.path.dirname(sys.executable)
        env_prefix_path = os.path.normpath(env_prefix) if env_prefix is not None else None
        normalized_env_prefix = os.path.normcase(env_prefix_path) if env_prefix_path else None
        normalized_executable_prefix = os.path.normcase(os.path.normpath(executable_prefix))
        if normalized_env_prefix and normalized_env_prefix != normalized_executable_prefix:
            return False, 'CONDA_PREFIX points to {}, but Python executable is under {}'.format(
                env_prefix_path,
                os.path.normpath(executable_prefix),
            )

        conda_prefix = executable_prefix
        conda_prefix = os.path.normpath(conda_prefix)

        expected_dirs = [
            conda_prefix,
            os.path.join(conda_prefix, 'Library', 'mingw-w64', 'bin'),
            os.path.join(conda_prefix, 'Library', 'usr', 'bin'),
            os.path.join(conda_prefix, 'Library', 'bin'),
            os.path.join(conda_prefix, 'Scripts'),
        ]
        path_dirs = [os.path.normpath(part) for part in os.environ.get('PATH', '').split(os.pathsep) if part]
        expected_head = [os.path.normcase(path) for path in expected_dirs]
        actual_head = [os.path.normcase(path) for path in path_dirs[:len(expected_dirs)]]
        if actual_head != expected_head:
            return False, 'PATH must start with {}'.format(expected_dirs[0])

        path_lookup = {os.path.normcase(path): index for index, path in enumerate(path_dirs)}

        missing = [path for path in expected_dirs if os.path.normcase(path) not in path_lookup]
        if missing:
            return False, 'missing {}'.format(missing[0])

        base_prefix = os.path.normpath(os.path.join(conda_prefix, '..', '..'))
        if os.path.basename(os.path.dirname(conda_prefix)).lower() == 'envs':
            base_dirs = [
                base_prefix,
                os.path.join(base_prefix, 'Library', 'mingw-w64', 'bin'),
                os.path.join(base_prefix, 'Library', 'usr', 'bin'),
                os.path.join(base_prefix, 'Library', 'bin'),
                os.path.join(base_prefix, 'Scripts'),
            ]
            base_indexes = [
                path_lookup[os.path.normcase(path)]
                for path in base_dirs
                if os.path.normcase(path) in path_lookup
            ]
            if base_indexes:
                first_base_index = min(base_indexes)
                late_expected = [
                    path for path in expected_dirs
                    if path_lookup[os.path.normcase(path)] > first_base_index
                ]
                if late_expected:
                    return False, '{} appears after base Anaconda PATH entries'.format(late_expected[0])

        return True, 'ok'
