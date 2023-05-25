import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
os.environ['MUJOCO_PY_MUJOCO_PATH']='/scratch_net/biwidl211/rl_course_10/.mujoco/mujoco210'

#add DiffuseDrive to path. Dont know why, but else diffuser folder is not seen.
# os.environ['PYTHONPATH'] = str(os.environ['PYTHONPATH']) + '~/DiffuseDrive'

import diffuser.utils as utils
import torch
import wandb
# logging.basicConfig(filename='logs/training.log', encoding='utf-8', level=logging.INFO)
from logger_module import logger

def main():

    class Parser(utils.Parser):
        dataset: str = 'carla-expert'
        config: str = 'config.carla'

    args = Parser().parse_args('diffusion')

    # start wandb logging
    
    if True:
        wandb_run = wandb.init(project="diffuse_drive", entity="mleong", config=args)


    torch.backends.cudnn.benchmark = True
    utils.set_seed(args.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        args.loader,
        savepath='dataset_config.pkl',
        env=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        include_returns=args.include_returns,
        returns_scale=args.returns_scale,
        discount=args.discount,
        termination_penalty=args.termination_penalty,
        past_image_cond = args.past_image_cond
    )

    render_config = utils.Config(
        args.renderer,
        savepath='render_config.pkl',
        env=args.dataset,
    )

    dataset = dataset_config()
    # renderer = render_config()
    renderer = None
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    if args.diffusion == 'models.GaussianInvDynDiffusion':
        # model_config = utils.Config(
        #     args.model,
        #     savepath='model_config.pkl',
        #     horizon=args.horizon,
        #     transition_dim=observation_dim,
        #     cond_dim=observation_dim,
        #     dim_mults=args.dim_mults,
        #     returns_condition=args.returns_condition,
        #     dim=args.dim,
        #     condition_dropout=args.condition_dropout,
        #     calc_energy=args.calc_energy,
        #     device=args.device,
        #     past_image_cond = args.past_image_cond,
        # )

        # diffusion_config = utils.Config(
        #     args.diffusion,
        #     savepath='diffusion_config.pkl',
        #     horizon=args.horizon,
        #     observation_dim=observation_dim,
        #     action_dim=action_dim,
        #     n_timesteps=args.n_diffusion_steps,
        #     loss_type=args.loss_type,
        #     clip_denoised=args.clip_denoised,
        #     predict_epsilon=args.predict_epsilon,
        #     hidden_dim=args.hidden_dim,
        #     ar_inv=args.ar_inv,
        #     train_only_inv=args.train_only_inv,
        #     ## loss weighting
        #     action_weight=args.action_weight,
        #     loss_weights=args.loss_weights,
        #     loss_discount=args.loss_discount,
        #     returns_condition=args.returns_condition,
        #     condition_guidance_w=args.condition_guidance_w,
        #     device=args.device,
        # )
        raise NotImplementedError
    else:
        model_config = utils.Config(
            args.model,
            savepath='model_config.pkl',
            horizon=args.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            dim=args.dim,
            condition_dropout=args.condition_dropout,
            calc_energy=args.calc_energy,
            device=args.device,
            past_image_cond = args.past_image_cond,
            resnet_freeze = args.resnet_freeze,
            # attention??
        )

        diffusion_config = utils.Config(
            args.diffusion,
            savepath='diffusion_config.pkl',
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            hidden_dim=args.hidden_dim,
            ar_inv=args.ar_inv,
            train_only_inv=args.train_only_inv,
            ## loss weighting
            action_weight=args.action_weight,
            loss_weights=args.loss_weights,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            device=args.device,
        )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        bucket=args.bucket,
        n_reference=args.n_reference,
        train_device=args.device,
        save_checkpoints=args.save_checkpoints,
    )
    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, renderer, wandb_run)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.info('Testing forward pass...')
    batch = utils.batchify(next(iter(dataset)), args.device)
    loss, _ = diffusion.loss(*batch)
    loss.backward()

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    wandb_run.config.update({"n_epochs": n_epochs})
    
    logger.info(f'Starting training for {n_epochs} epochs...')
    for i in range(n_epochs):
        trainer.train(n_train_steps=args.n_steps_per_epoch)

    wandb_run.finish()

if __name__ == '__main__':
    print("start main")
    main()
    