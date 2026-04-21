##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CSTPd_cluster.POMO.CTSPd_Trainer import TSPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'num_groups': 8,
    'relaxation_d': 1,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'num_groups': 8,
    'eval_type': 'argmax',
    'use_group_embedding': True,
    'use_group_fusion_gate': True,
    'cluster_bias_mode': 'scheduled',
    'same_group_bias_init': 0.1,
    'same_group_bias_final': 1.25,
    'same_group_bias_warmup_epochs': 20,
    'priority_distance_bias': 0.15,
    'priority_distance_tau': 1.0,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [80,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 160,
    'train_episodes': 100 * 1000,
    'train_batch_size': 256,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 5,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_ctspd_20.json',
            'title_prefix': 'CTSPd Cluster'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json',
            'title_prefix': 'CTSPd Cluster'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    },
    'early_stopping': {
        'enable': True,
        'monitor': 'train_score',
        'mode': 'min',
        'patience': 20,
        'min_delta': 1e-4,
        'warmup_epochs': 15,
        'checkpoint_best': True,
    },
}

logger_params = {
    'log_file': {
        'desc': 'cluster_n100_d1',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
