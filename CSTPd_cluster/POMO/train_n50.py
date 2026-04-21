##########################################################################################
# Machine Environment Config

import os
os.environ['MPLBACKEND'] = 'Agg'  # ← 必须放在第一位！在任何import之前

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.environ['MPLBACKEND'] = 'Agg'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
    'problem_size': 50,
    'pomo_size': 50,
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
    'same_group_bias_warmup_epochs': 40,
    'priority_distance_bias': 0.15,
    'priority_distance_tau': 1.0,
    # Enhanced cluster-aware switches. To reproduce the previous cluster model,
    # set relation_bias_mode='none' and use_decoder_priority_bias=False.
    'relation_bias_mode': 'learnable',
    'relation_bias_init': 0.2,
    'relation_bias_tau': 1.0,
    'use_decoder_priority_bias': True,
    'decoder_priority_bias_mode': 'learnable',
    'decoder_priority_bias_init': 0.2,
    'decoder_priority_bias_tau': 1.0,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [360,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 450,
    'train_episodes': 100 * 1000,
    'train_batch_size': 512,
    'logging': {
        'model_save_interval': 40,
        'img_save_interval': 20,
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
        # 'path': './result/saved_ctspd20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    },
    'early_stopping': {
        'enable': True,
        'monitor': 'train_score',
        'mode': 'min',
        'patience': 80,
        'min_delta': 1e-4,
        'warmup_epochs': 60,
        'checkpoint_best': True,
    },
}

logger_params = {
    'log_file': {
        'desc': 'train__cluster_ctspd_n50_g8_d1',
        'filename': 'run_log'
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
