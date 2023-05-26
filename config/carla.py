import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnetCarla',
        'diffusion': 'models.GaussianInvDynDiffusionCarla',
        'horizon': 12,
        'n_diffusion_steps': 200,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': True,
        'dim_mults': (1, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        "returns_condition": True,
        "calc_energy": False,
        "dim": 128,
        "condition_dropout": 0.25,
        "condition_guidance_w": 1.2,
        "test_ret": 0.9,
        "renderer": "utils.MuJoCoRenderer",
        "past_image_cond": False, 

        ## dataset
        "dataset": 'carla-expert',
        'loader': 'datasets.CollectedSequenceDataset',
        'normalizer': 'CDFNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        "include_returns": True,
        "discount": 0.99,
        "max_path_length": 10,
        "hidden_dim": 256,
        'ar_inv': False,
        "train_only_inv": False,
        "termination_penalty": -100,
        "returns_scale": 400.0,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 2500,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'resnet_freeze': False,
        'sample_freq': 100,
        'log_freq': 10,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'save_checkpoints': False,
        'bucket': None,
        'device': 'cuda',
        'seed': 100,
    },
    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
}
