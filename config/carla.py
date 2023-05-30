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
        'image_backbone': 'lraspp_mobilenet', # resnet18, lraspp_mobilenet
        'diffusion': 'models.GaussianInvDynDiffusionCarla',
        'horizon': 12,
        'n_diffusion_steps': 200,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 0.9,
        'predict_epsilon': True,
        'dim_mults': (1, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        "returns_condition": True,
        "calc_energy": False,
        "dim": 128,
        "condition_dropout": 0.01,
        "condition_guidance_w": 1.2,
        "test_ret": 0.9,
        "renderer": "utils.MuJoCoRenderer",
        "past_image_cond": True, 
        "checkpoint_path": 'TODO.pth.tar',

        ## dataset
        "dataset": 'carla-expert',
        'loader': 'datasets.CollectedSequenceDataset',
        'normalizer': 'CDFNormalizer',
        'waypoints_normalization': 'full_space', # 'full_space', 'single_waypoints', None
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
        'n_train_steps': 20000,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'lr_decay': 0.5,
        'lr_decay_steps': 2000,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'image_backbone_freeze': True,
        'sample_freq': 100,
        'log_freq': 10,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'save_checkpoints': False,
        'bucket': None,
        'device': 'cuda',
        'seed': 100,
        
        
        # planning with learned model on carla
        'turn_KP': 1.25,
        'turn_KI': 0.75,
        'turn_KD': 0.3,
        'turn_n': 40, # buffer size

        'speed_KP': 5.0,
        'speed_KI': 0.5,
        'speed_KD': 1.0,
        'speed_n': 40,  # buffer size

        'max_throttle': 0.75,  # upper limit on throttle signal value in dataset
        'brake_speed': 0.1,  # desired speed below which brake is triggered
        'brake_ratio': 1.1,  # ratio of speed to desired speed at which brake is triggered
        'clip_delta': 0.35,  # maximum change in speed input to logitudinal controller

        'skip_frames': 1,
    },

}
