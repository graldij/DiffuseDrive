import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
from logger_module import logger

from .arrays import batch_to_device, to_np, to_device, apply_dict, to_torch
from .timer import Timer
from .cloud import sync_logs
import diffuser.utils as utils
import matplotlib.pyplot as plt
from torchvision import transforms
import random
from PIL import Image
import pandas as pd

def cycle(dl):
    while True:
        for data in dl:
            yield data
def reset_seeds():
    # Set all seeds
    SEED = 0
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        valid_dataset,
        renderer,
        wandb_run=None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        lr_decay=0.9,
        lr_decay_steps = 1000,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=100,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        save_final=True,
        final_model_path = None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints
        
        self.save_final = save_final
        self.savepath = final_model_path

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.valid_dataset = valid_dataset

        # Shuffling is handled by sequence.py with the flag "is_valid". Here everythin should be set to shuffle = False
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=8, pin_memory=True, shuffle=False
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=1, num_workers=0, pin_memory=True
        ))
        self.renderer = renderer
        
        # optimizer and lr scheduler
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_steps, gamma=lr_decay)
        

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
        
        self.wandb_run = wandb_run

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
        
    def images_batch_norm(self, images):
        # Normalization from Resnet18. img_tmp loaded as PIL then converted to tensor. Therefore first map between 0 and 1, and then normalize
        normalization = transforms.Compose([transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                                            ])

        unsqueezed_images = normalization(images/255)
        
        return unsqueezed_images

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        # reset seeds here to make sure that DataLoading is reproducible
        reset_seeds()
        
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):

                batch = next(self.dataloader)
                # print("after load batch", timer(True))
                batch = batch_to_device(batch, device=self.device)
                # print("after batch to device", timer(True))
                
                # if conditioning on past images is True, then need to normalize the images
                if self.model.model.past_image_cond:
                    normalized_batch = self.images_batch_norm(batch.images)
                    new_batch = (batch.trajectories, batch.conditions, normalized_batch, batch.cmd) if self.model.model.using_cmd else (batch.trajectories, batch.conditions, normalized_batch)
                    batch = new_batch
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                # print("afterloss ", timer(True))

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
                
            if self.step % self.save_freq == 0 and self.save_checkpoints:
                self.save(savepath=self.savepath)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.info(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()

                if self.wandb_run is not None:
                    if self.sample_freq and self.step % self.sample_freq != 0:
                        self.wandb_run.log({**metrics, 'train/step': self.step, 'train/lr': self.scheduler.get_last_lr()[0]})
                    else:
                        self.wandb_run.log({**metrics, 'train/step': self.step, 'train/lr': self.scheduler.get_last_lr()[0]}, commit=False)
            
            # if self.step == 0 and self.sample_freq:
                # continue
                # [MOD] No rendering activated
                # TODO: add some kind of evaluation?
                # self.render_reference(self.n_reference)
            # self.step+=1
            # continue
            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
                    raise NotImplementedError
                    self.inv_render_samples()
                elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
                    raise NotImplementedError
                    pass
                else: # carla setting
                    # self.render_samples()
                    # sample and visualize with bird-eye-view
                    self.visualize_bev()
                    # Add evaluation accuracy
                    
            # step scheduler for lr decay
            self.scheduler.step()
            
            self.step += 1
            
        if self.save_final:
            self.save(savepath=self.savepath)

    def save(self, savepath=None):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        # savepath = os.path.join(self.bucket, 'checkpoint')
        savepath = savepath +'/'+ self.wandb_run.id
        
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'step_model_ema.pt')
        torch.save(data, savepath)
        logger.info(f'[ utils/training ] Saved model to {savepath}')

    def load(self, loadpath=None):
        '''
            loads model and ema from disk
        '''
        # loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        # [Mod] remove shuffle, same reason as above
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        observations = trajectories[:, :, self.dataset.action_dim:]
        # observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join('images', f'sample-reference.png')
        # self.renderer.composite(savepath, observations)

    # TODO Jacopo: remove hardcoded stuff and allow for different batch sizes and n_samples
    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            trajectories = to_device(batch.trajectories, self.device)
            conditions = to_device(batch.conditions, self.device)
            
            conditions = einops.repeat(conditions, 'b t d -> (repeat b) t d', repeat =n_samples)
            
            if hasattr(batch, "images") and self.ema_model.model.past_image_cond:
                # normalize images used as condition
                images = self.images_batch_norm(batch.images)
                
                images = to_device(images, self.device)
                images = einops.repeat(images, 'b t h w d -> (repeat b) t h w d', repeat = n_samples)
                if self.ema_model.model.using_cmd:
                    commands = batch.cmd
                    commands = to_device(commands, self.device)
                    samples = self.ema_model.conditional_sample(conditions, images=images, cmd = commands)
                else:
                    samples = self.ema_model.conditional_sample(conditions, images=images, cmd = None)
            else:
                if self.ema_model.model.using_cmd:
                    commands = batch.cmd
                    commands = to_device(commands, self.device)
                    samples = self.ema_model.conditional_sample(conditions, images=None, cmd = commands)
                else:
                    samples = self.ema_model.conditional_sample(conditions, images=None, cmd = None)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]
            
            # TODO Jacopo improve this
            # de-normalize observations
            traj_mean, traj_std = self.dataset.get_mean_std_waypoints()
            sampled_poses = normed_observations * (traj_std + 1e-7) + traj_mean
            true_trajectories = batch.trajectories * (traj_std + 1e-7) + traj_mean

            
            fig, ax = plt.subplots()
            
            
            colors = ['r', 'y']
            for samples in sampled_poses:
                color = colors.pop()
                for poses in samples:
                    
                    dx = np.cos(poses[2] - np.pi/2.0)
                    dy = np.sin(poses[2] - np.pi/2.0)
                    ax.arrow(poses[0], poses[1], dx, dy, head_width=0.00, head_length=0.0, color=color, alpha=0.5)
            for samples in true_trajectories:
                for poses in samples:
                    dx = np.cos(poses[2] - np.pi/2.0) 
                    dy = np.sin(poses[2] - np.pi/2.0)
                    ax.arrow(poses[0], poses[1], dx, dy, head_width=0.00, head_length=0.0, color='g', alpha=0.5)

            if not os.path.exists('plot/'+ self.wandb_run.id):
                os.makedirs('plot/'+ self.wandb_run.id)
            new_file_name = 'plot/'+ self.wandb_run.id +  '/result'+str(self.step) +"b" + str(i) +'.pdf'
            # plt.xlim([-10,10])
            # plt.autoscale_view()
            plt.autoscale(enable=True)
            plt.savefig(new_file_name)
            
            print("saved image", new_file_name)
            ax.cla()

    def visualize_bev(self, batch_size=2, n_samples=2, save_data=False):
        '''
            MOD Minxuan: Here I assume following:
            bev image is from the batch called batch.birdview
            we take two samples from the diffusion model
            The ground truth is in purple, random sample color for the sampled waypoints
        '''
        traj_mean, traj_std = self.dataset.get_mean_std_waypoints()
        l2_dist = [0] * traj_mean.shape[0]
        for i in range(batch_size):

            ## get a single datapoint
            
            # speed up 5x the visualization
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            batch = self.dataloader_vis.__next__()
            
            trajectories = to_device(batch.trajectories, self.device)
            conditions = to_device(batch.conditions, self.device)
            
            conditions = einops.repeat(conditions, 'b t d -> (repeat b) t d', repeat =n_samples)
            
            if hasattr(batch, "images") and self.ema_model.model.past_image_cond:
                # normalize images used as condition
                images = self.images_batch_norm(batch.images)
                
                images = to_device(images, self.device)
                images = einops.repeat(images, 'b t h w d -> (repeat b) t h w d', repeat = n_samples)
                if self.ema_model.model.using_cmd:
                    commands = batch.cmd
                    commands = to_device(commands, self.device)
                    commands = einops.repeat(commands, 'b t h  -> (repeat b) t h ', repeat = n_samples)
                    samples = self.ema_model.conditional_sample(conditions, images=images, cmd = commands)
                else:
                    samples = self.ema_model.conditional_sample(conditions, images=images, cmd = None)
            else:
                if self.ema_model.model.using_cmd:
                    commands = batch.cmd
                    commands = to_device(commands, self.device)
                    commands = einops.repeat(commands, 'b t h  -> (repeat b) t h ', repeat = n_samples)
                    samples = self.ema_model.conditional_sample(conditions, images=None, cmd = commands)
                else:
                    samples = self.ema_model.conditional_sample(conditions, images=None, cmd = None)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]
            
            # TODO Jacopo improve this
            # de-normalize observations
            traj_min, traj_max = self.dataset.get_min_max_waypoints()
            # sampled_poses = 0.5* (normed_observations+1.0) * (traj_max-traj_min) + traj_min 
            # true_trajectories = 0.5* (batch.trajectories+1.0) * (traj_max-traj_min) + traj_min 
            sampled_poses = normed_observations
            true_trajectories = batch.trajectories

            traj_mean, traj_std = self.dataset.get_mean_std_waypoints()
            sampled_poses = sampled_poses * (traj_std + 1e-7) + traj_mean
            true_trajectories = true_trajectories * (traj_std + 1e-7) + traj_mean

            print(batch.trajectories)
            
            for j, observation in enumerate(normed_observations):
                print(observation)
                for k, waypoint in enumerate(observation):
                    l2_dist[k] += np.linalg.norm(waypoint - np.array(batch.trajectories[0][k]))
            # TODO might add a check if the birdview exists
            
            fig, ax = plt.subplots()
            ## plot bev image, hardcode it into 500*500
            plt.rcParams["figure.figsize"] = (6,6)
            margin_max = 400 
            margin_min = 0 
            ax.set_xlim(margin_min, margin_max)
            ax.set_ylim(margin_min, margin_max)
            bev_image = batch.birdview.numpy()
            img = Image.fromarray(bev_image.squeeze(), 'RGB')
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.resize((500,500))
            ax.imshow(img, extent=[0,500,0,500])
            
            ## scale for coloring
            length = true_trajectories.shape[1]*2
            for sample_pose in sampled_poses:
                c = np.random.rand(3,)
                for j, poses in enumerate(sample_pose):
                    dx = np.cos(poses[-1]-np.pi/2)/5
                    dy = np.sin(poses[-1]-np.pi/2)/5
                    
                    #waypoint = np.around(poses*5+20*10).astype(int)
                    # Hardcode it, scale=5
                    waypoint = poses*5+250
                    ax.scatter(waypoint[0], waypoint[1],s=10, color=c, alpha=0.5+j/length)
                    if j == 11:
                        ax.arrow(waypoint[0], waypoint[1], dx, dy, head_width=7, color='blue',alpha=0.5+j/length)
                    ## c represents current, should overlapping of all trajectories
                    if j== 3:
                        ax.text(waypoint[0]-0.05, waypoint[1]+0.05, "c", fontsize=10)

            for j, poses in enumerate(true_trajectories.squeeze()):
                    dx = np.cos(poses[-1]-np.pi/2)/5
                    dy = np.sin(poses[-1]-np.pi/2)/5
                    
                    #waypoint = np.around(poses*5+20*10).astype(int)
                    # Hardcode it, scale=5
                    waypoint = poses*5+250
                    ax.scatter(waypoint[0], waypoint[1],s=10, color='purple', alpha=0.5+j/length)
                    if j == 11:
                        ax.arrow(waypoint[0], waypoint[1], dx, dy, head_width=7, color='blue',alpha=0.5+j/length)
                    ## c represents current, should overlapping of all trajectories
                    if j == 3:
                        ax.text(waypoint[0]-0.05, waypoint[1]+0.05, "c", fontsize=10)
            ## save plot
            if not os.path.exists('visualize_bev/'+ self.wandb_run.id):
                os.makedirs('visualize_bev/'+ self.wandb_run.id)
            if self.ema_model.model.using_cmd:
                new_file_name = 'visualize_bev/'+ self.wandb_run.id +  '/result'+str(self.step) +"b" + str(i) + "cmd" + str(int(batch.cmd[0][3].argmax())) + '.png'
            else:
                new_file_name = 'visualize_bev/'+ self.wandb_run.id +  '/result'+str(self.step) +"b" + str(i) + '.png'
            plt.savefig(new_file_name)
            
            print("saved visualization", new_file_name)
            ax.cla()

            ## save data as csv file
            if save_data:
                col_name = []
                full_data = []
                row_data = np.array([])
                for time_step in range(batch.trajectories.shape[1]):
                    col_name.append('x%i' %time_step)
                    col_name.append('y%i' %time_step)
                    col_name.append('t%i' %time_step)
                    row_data = np.append(row_data, np.array(np.array(batch.trajectories[0][time_step])))
                full_data.append(row_data)

                for j, observation in enumerate(normed_observations):
                    row_data = np.array([])
                    for k, waypoint in enumerate(observation):
                        row_data = np.append(row_data, waypoint)
                    full_data.append(row_data)
                df = pd.DataFrame(i for i in full_data)
                df.columns = col_name
                df.to_csv('visualize_bev/' + self.wandb_run.id + '/result' + str(self.step) + "b" + str(i) + '.csv')



        for t,dist in enumerate(l2_dist):
            dist = dist / (batch_size * n_samples)
            if t < len(l2_dist) - 1:
                self.wandb_run.log({'eval/dist'+str(t): dist}, commit=False)
            else:
                self.wandb_run.log({'eval/dist'+str(t): dist})
            

    def inv_render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        raise NotImplementedError
        # for i in range(batch_size):

        #     ## get a single datapoint
        #     batch = self.dataloader_vis.__next__()
        #     conditions = to_device(batch.conditions, self.device)
        #     ## repeat each item in conditions `n_samples` times
        #     conditions = apply_dict(
        #         einops.repeat,
        #         conditions,
        #         'b d -> (repeat b) d', repeat=n_samples,
        #     )

        #     ## [ n_samples x horizon x (action_dim + observation_dim) ]
        #     if self.ema_model.returns_condition:
        #         returns = to_device(torch.ones(n_samples, 1), self.device)
        #     else:
        #         returns = None

        #     if self.ema_model.model.calc_energy:
        #         samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
        #     else:
        #         samples = self.ema_model.conditional_sample(conditions, returns=returns)

        #     samples = to_np(samples)

        #     ## [ n_samples x horizon x observation_dim ]
        #     normed_observations = samples[:, :, :]

        #     # [ 1 x 1 x observation_dim ]
        #     normed_conditions = to_np(batch.conditions[0])[:,None]

        #     # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        #     # observations = conditions + blocks_cumsum_quat(deltas)
        #     # observations = conditions + deltas.cumsum(axis=1)

        #     ## [ n_samples x (horizon + 1) x observation_dim ]
        #     normed_observations = np.concatenate([
        #         np.repeat(normed_conditions, n_samples, axis=0),
        #         normed_observations
        #     ], axis=1)

        #     ## [ n_samples x (horizon + 1) x observation_dim ]
        #     observations = normed_observations
        #     print(observations)
        #     # observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        #     #### @TODO: remove block-stacking specific stuff
        #     # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        #     # observations = blocks_add_kuka(observations)
        #     ####

        #     savepath = os.path.join('images', f'sample-{i}.png')
        #     self.renderer.composite(savepath, observations)