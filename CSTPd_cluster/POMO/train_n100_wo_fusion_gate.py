import train_n100 as base


base.model_params.update({
    'use_group_embedding': True,
    'use_group_fusion_gate': False,
    'cluster_bias_mode': 'learnable',
    'same_group_bias_init': 0.1,
    'same_group_bias_max': 2.0,
    'priority_distance_bias': 0.0,
    'priority_distance_tau': 1.0,
    'relation_bias_mode': 'learnable',
    'relation_bias_init': 0.2,
    'relation_bias_tau': 1.0,
    'use_decoder_priority_bias': False,
})

base.trainer_params['logging']['log_image_params_1']['title_prefix'] = 'CTSPd Cluster Learnable Bias w/o Fusion Gate'
base.trainer_params['logging']['log_image_params_2']['title_prefix'] = 'CTSPd Cluster Learnable Bias w/o Fusion Gate'
base.logger_params['log_file']['desc'] = 'cluster_n100_d1_learnable_bias_wo_fusion_gate'


if __name__ == "__main__":
    base.main()
