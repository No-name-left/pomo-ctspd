import train_n100 as base


base.model_params.update({
    'use_group_embedding': True,
    'use_group_fusion_gate': False,
    'cluster_bias_mode': 'scheduled',
    'priority_distance_bias': 0.15,
})

base.trainer_params['logging']['log_image_params_1']['title_prefix'] = 'CTSPd Cluster w/o Fusion Gate'
base.trainer_params['logging']['log_image_params_2']['title_prefix'] = 'CTSPd Cluster w/o Fusion Gate'
base.logger_params['log_file']['desc'] = 'train__cluster_ctspd_n100_g8_d1_wo_fusion_gate__160epoch_bs256'


if __name__ == "__main__":
    base.main()
