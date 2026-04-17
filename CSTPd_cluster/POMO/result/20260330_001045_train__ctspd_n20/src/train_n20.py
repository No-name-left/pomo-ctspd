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
from utils import create_logger, copy_all_src

from CTSPd_Trainer import TSPTrainer as Trainer
from CTSPd_Env import CTSPdEnv as Env
from CTSPd_Model import CTSPdModel as Model


##########################################################################################
# parameters

env_params = {
    'problem_size': 20,
    'pomo_size': 20,
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
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [501,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 200,
    'train_episodes': 100 * 1000,
    'train_batch_size': 512,
    'logging': {
        'model_save_interval': 40,
        'img_save_interval': 20,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_ctspd_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_ctspd20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    }
}

logger_params = {
    'log_file': {
        'desc': 'train__ctspd_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
# main
import torch
import torch.nn as nn

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # 临时创建对象进行检查
    temp_env = Env(**env_params)
    temp_model = Model(**model_params)
    
    if not check_setup(env_params, model_params, temp_model):
        print(" 配置检查失败，停止训练")
        print(f"请检查: num_groups 环境={env_params.get('num_groups')} vs 模型={model_params.get('num_groups')}")
        return  # 直接退出，不创建 trainer
    
    print("配置检查通过")
    # 【插入到这里结束】

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

def check_setup(env_params, model_params, model):
    """极简检查"""
    print("\n=== 快速检查 ===")
    
    # 检查 num_groups 是否一致
    ng_env = env_params.get('num_groups', 5)
    ng_model = model_params.get('num_groups', 5)
    
    print(f"num_groups: 环境={ng_env}, 模型={ng_model}")
    
    if ng_env != ng_model:
        print("错误: 不一致!会导致CUDA错误")
        return False
    
    # 检查聚类模块
    has_cluster = hasattr(model.encoder, 'group_embedding')
    print(f"聚类模块: {'✓' if has_cluster else '✗'}")
    
    if has_cluster:
        size = model.encoder.group_embedding.num_embeddings
        print(f"支持组数: {size-1} (索引1-{size-1})")
        if size < ng_env + 1:
            print(f"错误: Embedding太小({size})，需要{ng_env+1}")
            return False
    
    print("================\n")
    return True

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
