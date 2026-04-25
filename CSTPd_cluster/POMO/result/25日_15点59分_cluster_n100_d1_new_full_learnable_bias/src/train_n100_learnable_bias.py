import train_n100 as base


base.model_params.update({
    'use_group_embedding': True,
    'use_group_fusion_gate': True,
    # Learnable-bias full model initialized from the old full model's bias
    # schedule. With zero residuals this is exactly the old scheduled/fixed-bias
    # full model; training learns only deviations from that bias prior.
    'cluster_bias_mode': 'scheduled_residual',
    'same_group_bias_init': 0.1,
    'same_group_bias_final': 1.25,
    'same_group_bias_warmup_epochs': 20,
    'same_group_bias_max': 2.0,
    'priority_distance_bias': 0.15,
    'priority_distance_tau': 1.0,
    'relation_bias_mode': 'learnable',
    'relation_bias_init': 0.0,
    'relation_bias_tau': 1.0,
    'learnable_bias_start_epoch': 1,
    'learnable_bias_warmup_epochs': 1,
    'learnable_bias_scale_max': 1.0,
    'use_decoder_priority_bias': False,
})

base.trainer_params['logging']['log_image_params_1']['title_prefix'] = 'CTSPd Cluster New Full Learnable Bias'
base.trainer_params['logging']['log_image_params_2']['title_prefix'] = 'CTSPd Cluster New Full Learnable Bias'
base.logger_params['log_file']['desc'] = 'cluster_n100_d1_new_full_learnable_bias'


if __name__ == "__main__":
    base.main()
