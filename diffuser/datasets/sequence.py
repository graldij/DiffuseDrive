from collections import namedtuple
from unicodedata import name
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from .carla_dataset.carla_dataset import CarlaDataset
from datasets import load_dataset
from torchvision import transforms

RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
# [Mod] add new namedtuple for our case
ImageBatch = namedtuple('Batch', 'trajectories conditions images')
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
# MOD Minxuan: add two tuples for validation dataset with/without image conditioning
ValidBatch = namedtuple('ValidBatch', 'trajectories conditions birdview')
ValidImageBatch = namedtuple('ValidImageBatch', 'trajectories conditions images birdview')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        # initialize buffer with zeros
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        # remove episodes that are not filled
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths']) # normalize all the data input
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        # (100,11)
        observations = self.fields.normed_observations[path_ind, start:end]
        #(100,3)
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class CondSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        t_step = np.random.randint(0, self.horizon)

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        traj_dim = self.action_dim + self.observation_dim

        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1

        if t_step < self.horizon-1:
            observations[t_step+1:] = 0

        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)
            batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch

# [Mod] created an Iterable dataset for our project, not tested what happens after the data is fully loaded
class CollectedSequenceDataset(torch.utils.data.IterableDataset):

    def __init__(self, env='carla-expert', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, discount=0.99, returns_scale=1000, include_returns=False, past_image_cond = True, waypoints_normalization = None, is_valid = False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        # self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        # itr = sequence_dataset(env, self.preprocess_fn)
        self.past_image_cond = past_image_cond
        self.waypoints_normalization = waypoints_normalization
        
        if past_image_cond:
            self.dataset = load_dataset("diffuser/datasets/carla_dataset", "decdiff", is_valid = is_valid, streaming=True, split="train")
        else:
            self.dataset = load_dataset("diffuser/datasets/carla_dataset", "waypoint_unconditioned", is_valid = is_valid, streaming=True, split="train")
        
        ## MOD Minxuan, add is_valid
        self.is_valid = is_valid
        ## not shuffle for validation set, perhaps could be comment
        if not self.is_valid:
            self.dataset.shuffle(seed=42, buffer_size=50)
        self.img_size = 128
        self.preprocess = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])

        # fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        # for i, episode in enumerate(itr):
        #     fields.add_path(episode)
        # fields.finalize()

        # self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        # self.indices = self.make_indices(fields.path_lengths, horizon)

        # self.observation_dim = fields.observations.shape[-1]
        # self.action_dim = fields.actions.shape[-1]
        # self.fields = fields
        # self.n_episodes = fields.n_episodes
        # self.path_lengths = fields.path_lengths
        # self.normalize()

        # print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')
        # TODO: change to actual data loaded, add some preprocessing, normalizer?
        self.action_dim = 0
        self.observation_dim = 3
        self.indices = [1] * 10

    # Currently normalize, make_indices, __len__ and get_conditions are not used
    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    
    def get_mean_std_waypoints(self):
        traj_mean, traj_std = 0.0, 1.0
        if self.waypoints_normalization == "single_waypoints":
            traj_mean = np.array([-1.4380759e-02, -4.2510300e+00, 1.1896066e-03])
            traj_std = np.array([1.3787538, 7.970438, 0.19030738])
        elif self.waypoints_normalization == "full_space":
            traj_mean = np.array([  [-9.6931e-03,  4.8700e+00, -1.4568e-03],
                                    [-4.6584e-03,  3.2597e+00, -9.3381e-04],
                                    [-1.6569e-03,  1.6339e+00, -4.6059e-04],
                                    [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                                    [ 6.8762e-04, -1.6337e+00,  4.3586e-04],
                                    [-5.8059e-05, -3.2622e+00,  8.5435e-04],
                                    [-2.9566e-03, -4.8838e+00,  1.2646e-03],
                                    [-8.2530e-03, -6.4950e+00,  1.6922e-03],
                                    [-1.5935e-02, -8.0928e+00,  2.1226e-03],
                                    [-2.5734e-02, -9.6753e+00,  2.5433e-03],
                                    [-3.7314e-02, -1.1242e+01,  2.9973e-03],
                                    [-5.0312e-02, -1.2791e+01,  3.4961e-03]])
            traj_std = np.array([   [0.3209, 4.0127, 0.1432],
                                    [0.1167, 2.7227, 0.0992],
                                    [0.0225, 1.3844, 0.0514],
                                    [0.0000, 0.0000, 0.0000],
                                    [0.1156, 1.3830, 0.0515],
                                    [0.3243, 2.7180, 0.0999],
                                    [0.6107, 4.0036, 0.1449],
                                    [0.9655, 5.2458, 0.1867],
                                    [1.3822, 6.4494, 0.2254],
                                    [1.8558, 7.6204, 0.2616],
                                    [2.3817, 8.7654, 0.2957],
                                    [2.9568, 9.8893, 0.3280]])
        elif self.waypoints_normalization == None:
            return traj_mean, traj_std
        else:
            raise NotImplementedError
        
        return traj_mean, traj_std
        
        

    def __len__(self):
        return len(self.indices)

    # This part of the code is ran when loading data
    def __iter__(self):
        # path_ind, start, end = self.indices[idx]

        # observations = self.fields.normed_observations[path_ind, start:end]
        # actions = self.fields.normed_actions[path_ind, start:end]

        for i in self.dataset.iter(1):
            actions = i["actions"]
            trajectories = np.array(actions).squeeze(0)

            ## MOD Minxuan: add birdview for validation dataset
            if self.is_valid:
                bev_img = i["birdview"]
            # filter out the trajectories where the car is not moving, i.e. the maximum values in the horizon (future or past) are close to 0
            if np.absolute(trajectories[:,:-1]).max() <= 1e-6:
                continue
            else:
                # if the car is not turning, we keep the sample with probability 0.1 and discard it with probability 0.9. 
                # Car is not turning if the max value of the x direction is less than 10/5 (/5 coming from the normalization and visualization with bev)
                if np.absolute(trajectories[:,0]).max() <= 10/5:
                    # generate a random boolean variable with probability of beging true of 0.1
                    keep_sample = np.random.choice([True, False], p=[0.01, 0.99])
                    if not keep_sample:
                        continue
                # Normalize waypoints
                traj_mean, traj_std = self.get_mean_std_waypoints()
                trajectories = (trajectories - traj_mean)/(traj_std + 1e-7)
                
                conditions = trajectories[:4, :].copy()
                
                if self.past_image_cond:
                    observations = i["observations"]
                    image = np.zeros((len(observations[0]), 3, self.img_size, self.img_size))
                    for t, img_temp in enumerate(observations[0][:]):
                        unsqueezed_image = np.array(img_temp)[np.newaxis, :].transpose((0,3,1,2))
                        image[t] = unsqueezed_image
                    
                    ## MOD Minxuan: add option for validation dataset, both for conditioning & unconditioning
                    if self.is_valid:
                        batch = ValidImageBatch(trajectories.astype(np.float32), conditions.astype(np.float32), image.astype(np.float32), np.asarray(bev_img[0]))
                    else:
                        batch = ImageBatch(trajectories.astype(np.float32), conditions.astype(np.float32), image.astype(np.float32))
                else:
                    if self.is_valid:
                        batch = ValidBatch(trajectories.astype(np.float32), conditions.astype(np.float32), np.asarray(bev_img[0]))
                    else:
                        batch = Batch(trajectories.astype(np.float32), conditions.astype(np.float32))
                yield batch