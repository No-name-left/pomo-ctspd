import train_n100 as base


base.model_params.update({
    'use_group_embedding': True,
    'use_group_fusion_gate': True,
    'cluster_bias_mode': 'none',
    'priority_distance_bias': 0.15,
})

base.trainer_params['logging']['log_image_params_1']['title_prefix'] = 'CTSPd Cluster w/o Cluster Bias'
base.trainer_params['logging']['log_image_params_2']['title_prefix'] = 'CTSPd Cluster w/o Cluster Bias'
base.logger_params['log_file']['desc'] = 'cluster_n100_d1_wo_cluster_bias'


if __name__ == "__main__":
    base.main()
