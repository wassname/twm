CONFIGS = {}

CONFIGS["default"] = {
    # buffer
    "buffer_capacity": 200_000, # 200k, ~8GB
    "buffer_temperature": 20.0,
    "buffer_prefill": 20_000, # need to be proportional to batch size

    # training
    "train_it_budget": 3_000_000,  # 100k in 10hrs. this seems to be wm+ac training steps. was 1_000_000_000. # 3h, and b=800, 1M
    "pretrain_it_budget": 20_000_000,
    "pretrain_obs_p": 0.4,
    "pretrain_dyn_p": 0.6,

    # evaluation
    "save_every": 100_000,
    "eval_every": 15000,
    "eval_episodes": 10,
    "final_eval_episodes": 100,

    # environment
    "env_step_budget": 1_000_000, #  was 100k for breakout, 1M or 1B for crafter
    "env_frame_size": 8268, # craftax
    "env_frame_stack": 2,
    "env_time_limit": 10_000, # only during training
    "env_discount_factor": 0.99,
    "env_discount_lambda": 0.95,

    # world model
    "wm_batch_size": 800, # 17GB
    "wm_sequence_length": 16,
    "wm_train_steps": 1,
    "wm_memory_length": 16,
    "wm_discount_threshold": 0.1,

    "z_categoricals": 256,
    "z_categories": 256,

    # "obs_channels": 48,
    "obs_act": "silu",
    "obs_norm": "none",
    "obs_dropout": 0,
    "obs_lr": 1e-4,
    "obs_wd": 1e-6,
    "obs_eps": 1e-5,
    "obs_grad_clip": 100,
    "obs_entropy_coef": 5,
    "obs_entropy_threshold": 0.1,
    "obs_consistency_coef": 0.01,
    "obs_decoder_coef": 1,

    "dyn_embed_dim": 256,
    "dyn_num_heads": 4,
    "dyn_num_layers": 10,
    "dyn_feedforward_dim": 1024,
    "dyn_head_dim": 64,
    "dyn_z_dims": [512, 512, 512, 512],
    "dyn_reward_dims": [256, 256, 256, 256],
    "dyn_discount_dims": [256, 256, 256, 256],
    "dyn_input_rewards": True,
    "dyn_input_discounts": False,
    "dyn_act": "silu",
    "dyn_norm": "none",
    "dyn_dropout": 0.1,
    "dyn_lr": 1e-4,
    "dyn_wd": 1e-6,
    "dyn_eps": 1e-5,
    "dyn_grad_clip": 100,
    "dyn_z_coef": 1,
    "dyn_reward_coef": 10,
    "dyn_discount_coef": 50,

    # actor-critic
    "ac_batch_size": 800,
    "ac_horizon": 15,
    "ac_act": "silu",
    "ac_norm": "none",
    "ac_dropout": 0,
    "ac_input_h": False,
    "ac_h_norm": "none",
    "ac_normalize_advantages": False,
    "actor_dims": [512, 512, 512, 512],
    "actor_lr": 1e-4,
    "actor_eps": 1e-5,
    "actor_wd": 1e-6,
    "actor_entropy_coef": 1e-2,
    "actor_entropy_threshold": 0.1,
    "actor_grad_clip": 1,

    "critic_dims": [512, 512, 512, 512],
    "critic_lr": 1e-5,
    "critic_eps": 1e-5,
    "critic_wd": 1e-6,
    "critic_grad_clip": 1,
    "critic_target_interval": 1,
}

CONFIGS["test"] = {
    **CONFIGS["default"],
    # smallest possible run that will test batch size, train, eval, and buffer wrap around
    "buffer_capacity": 15_000,
    "buffer_prefill": 13_000,
    "train_it_budget": 50_000,
    "pretrain_it_budget": 5_000,
    "eval_every": 25_000,
    "save_every": 10_000,
    "env_time_limit": 1870,
    # "wm_batch_size": 8,
    # "ac_batch_size": 8,
    # "wm_sequence_length": 8,
    "env_step_budget": 73_000,
}
