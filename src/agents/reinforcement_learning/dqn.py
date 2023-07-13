"""
DQN Implementation with target net and epsilon greedy. Follows the Stable Baselines 3 implementation.
To reuse trained models, you can make use of the save and load function.
To adapt policy and value network structure, specify the layer and activation parameter in your train config or
change the constants in this file
"""
import numpy as np
import pickle
import random
from collections import deque
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from typing import Tuple, List

from torch.autograd import Variable
from src.agents.reinforcement_learning.sumtree import SumTree

# 추후 삭제
import time
import warnings

from src.utils.logger import Logger

# constants
LAYER: List[int] = [64, 64]
ACTIVATION: str = 'ReLU'


class MemoryBuffer:
    """
    Handles episode data collection and sample generation

    :param buffer_size: Buffer size
    :param batch_size: Size for batches to be generated
    :param obs_dim: Size of the observation to be stored in the buffer
    :param obs_type: Type of the observation to be stored in the buffer
    :param action_type: Type of the action to be stored in the buffer

    """

    # PER 위해 추가
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    ####

    def __init__(self, buffer_size: int, batch_size: int, obs_dim: int, obs_type: type, action_type: type):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.pos = 0
        self.full = False

        # buffer data
        self.obs = np.zeros((buffer_size, obs_dim), dtype=obs_type)
        self.actions = np.zeros((buffer_size, 1), dtype=action_type)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)   # try with   dtype=np.bool
        self.new_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)

        # SumTree 추가
        self.tree = SumTree(buffer_size)

    # PER 위해 추가
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.pos

    # 기존 방식
    """    
    def store_memory(self, obs, action, reward, done, new_obs) -> None:

        self.obs[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.new_obs[self.pos] = np.array(new_obs)

        self.pos += 1
        # if pos behind last element -> buffer full.
        # Return pos to 0. Next step, the oldest data in the buffer is then replaced by the newest one
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    """

    # PER에 맞게 수정한 방식
    def store_memory(self, error, obs, action, reward, done, new_obs) -> None:
        
        sample = (obs, action, reward, done, new_obs)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    # 기존 방식
    """
    def get_samples(self) -> Tuple:

        indices = np.random.randint(0, len(self), size=self.batch_size)

        return self.obs[indices], self.actions[indices], self.rewards[indices], \
            self.dones[indices], self.new_obs[indices]
    """

    # PER에 맞게 수정한 방식
    def get_samples(self):

        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        epsilon = 1e-10  # very small value to avoid division by zero
        sampling_probabilities = np.where(sampling_probabilities == 0, epsilon, sampling_probabilities)

        '''
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        except RuntimeWarning:
            print(f"sampling_probabilities = {sampling_probabilities}")
            print(f"n_entries = {self.tree.n_entries}")
            print(f"beta = {self.beta}")
            time.sleep(20)
        '''

        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight
    
    # PER 적용 위해 추가
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class Policy(nn.Module):
    """
    Network structure used for both the Q network and the target network

    :param obs_dim: Observation size to determine input dimension
    :param action_dim: Number of action to determine output size
    :param learning_rate: Learning rate for the network
    :param hidden_layers: List of hidden layer sizes (int)
    :param activation: String naming activation function for hidden layers

    """
    def __init__(self, obs_dim: int, action_dim: int, learning_rate: float, hidden_layers: List[int], activation: str):
        super(Policy, self).__init__()

        net_structure = []
        # get activation class according to string
        activation = getattr(nn, activation)()

        # create first hidden layer in accordance with the input dim and the first hidden dim
        net_structure.extend([nn.Linear(obs_dim, hidden_layers[0]), activation])

        # create the other hidden layers
        for i, layer_dim in enumerate(hidden_layers):
            if not i + 1 == len(hidden_layers):
                net_structure.extend([nn.Linear(layer_dim, hidden_layers[i + 1]), activation])
            else:
                # create output layer
                net_structure.append(nn.Linear(layer_dim, action_dim))

        self.q_net = nn.Sequential(*net_structure)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        ####################################################################################
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        ####################################################################################
        self.to(self.device)

    def forward(self, obs):
        """ forward pass through the Q-network """
        q_values = self.q_net(obs)
        return q_values


class DQN:
    """DQN Implementation with target net and epsilon greedy. Follows the Stable Baselines 3 implementation."""
    def __init__(self, env, config: dict, logger: Logger = None):

        """
        | batch_size: Number of samples that are chosen and passed through the net per update
        | gradient_steps: Number of updates per training
        | train_freq: Environment steps between two trainings
        | buffer_size: Size of the memory buffer = max number of rollouts that can be stored before the oldest are deleted
        | target_net_update: Number of steps between target_net_updates
        | training_starts = Learning_starts: steps after which training can start for the first time
        | initial_eps: Initial epsilon value
        | final_eps: Final epsilon value
        | fraction_eps: If the percentage progress of learn exceeds fraction eps, epsilon takes the final_eps value
        | e.g. 5/100 total_timesteps done -> progress = 0.5 > fraction eps -> eps=final_eps
        | max_grad_norm: Value to clip the policy update of the q_net

        :param env: Pregenerated, gymbased environment. If no env is passed, env = None -> PPO can only be used
            for evaluation (action prediction)
        :param config: Dictionary with parameters to specify DQN attributes
        :param logger: Logger

        """

        self.env = env
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 32)
        self.gradient_steps = config.get('gradient_steps', 1)
        self.train_freq = config.get('train_freq', 4)
        self.buffer_size = config.get('buffer_size', 1_000_000)

        self.target_net_update = config.get('target_net_update', 10_000)
        self.training_starts = config.get('training_starts', 50_000)
        self.initial_eps = config.get('initial_eps', 1.0)
        self.final_eps = config.get('final_eps', 0.05)
        self.fraction_eps = config.get('fraction_eps', 0.1)
        self.max_grad_norm = config.get('max_grad_norm', 10.0)
        self.epsilon = self.initial_eps  # epsilon is the exploration rate
        self.remaining_progress = 1  # tracks how much % of total steps remain -> value between 1 and 0
        self.num_timesteps = 0
        self.n_updates = 0

        self.logger = logger if logger else Logger(config=config)
        self.seed = config.get('seed', None)
        self.reward_info = deque(maxlen=100)

        # torch seed setting
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            T.manual_seed(self.seed)
            self.env.action_space.seed(self.seed)
            self.env.seed(self.seed)

        # create networks and buffer
        self.q_net = Policy(env.observation_space.shape[0], env.action_space.n, self.learning_rate,
                            config.get('layer', LAYER),
                            config.get('activation', ACTIVATION))
        self.q_target_net = Policy(env.observation_space.shape[0], env.action_space.n, self.learning_rate,
                                   config.get('layer', LAYER),
                                   config.get('activation', ACTIVATION))
        # copy weights to target_net
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.memory_buffer = MemoryBuffer(self.buffer_size, self.batch_size, env.observation_space.shape[0],
                                          env.observation_space.dtype, env.action_space.dtype)
        
    def save(self, file: str) -> None:
        """
        Save model as pickle file

        :param file: Path under which the file will be saved

        :return: None

        """
        params_dict = self.__dict__.copy()
        del params_dict['logger']
        data = {
            "params": params_dict,
            "q_params": self.q_net.state_dict(),
            "target_params": self.q_target_net.state_dict()
        }

        with open(f"{file}.pkl", "wb") as handle:
            pickle.dump(data, handle)

    @classmethod
    def load(cls, file: str, config: dict, logger: Logger = None):
        """
        Creates a DQN object according to the parameters saved in file.pkl

        :param file: Path and filname (without .pkl) of your saved model pickle file
        :param config: Dictionary with parameters to specify PPO attributes
        :param logger: Logger

        :return: DQN object

        """
        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)

        env = data["params"]["env"]

        # create DQN object. Commit necessary parameters. Update remaining parameters
        model = cls(env=env, config=config, logger=logger)
        model.__dict__.update(data["params"])

        # set weights
        model.q_net.load_state_dict(data["q_params"])
        model.q_target_net.load_state_dict(data["target_params"])

        return model

    def get_action(self, obs: np.ndarray) -> int:
        """
        Random action or action according to the policy and epsilon

        :return: action index

        """
        if np.random.random() < self.epsilon:
            # random action from the action space
            action = self.env.action_space.sample()
        else:
            obs = T.tensor(obs, dtype=T.float).to(self.q_net.device)
            q_values = self.q_net(obs)
            # choose action with highest Q value -> greedy policy
            action = T.argmax(q_values)
            action = T.squeeze(action).item()

        return action

    def predict(self, observation: np.ndarray, action_mask: np.ndarray = np.ones(1),
                deterministic: bool = True, state=None) -> Tuple:
        """
        Action prediction for testing

        :param observation: Current observation of teh environment
        :param action_mask: Mask of actions, which can logically be taken. NOTE: currently not implemented!
        :param deterministic: Set True, to force a deterministic prediction
        :param state: The last states (used in rnn policies)

        :return: Predicted action and next state (used in rnn policies)

        """
        observation = T.tensor(np.array([observation]), dtype=T.float).to(self.q_net.device)

        with T.no_grad():
            q_values = self.q_net(observation)
            if deterministic:
                action = T.argmax(q_values)
            else:
                # choose random action according to the predicted probs
                action = q_values.sample()
            action = T.squeeze(action).item()

        return action, state

    def train(self) -> None:
        """
        Trains Q-network and Target-Network

        :return: None

        """
        # Switch to train mode (this affects batch norm / dropout)
        self.q_net.train()

        losses = []

        for _ in range(self.gradient_steps):

            # get samples from the buffer
            # 기존
            # obs_arr, action_arr, reward_arr, done_array, new_obs_array = self.memory_buffer.get_samples()
            # 수정
            is_print = self.memory_buffer.tree.total() > 147028
            batch, idxs, is_weights = self.memory_buffer.get_samples()

            obs_arr = []
            action_arr = []
            reward_arr = []
            done_array = []
            new_obs_array = []
            
            # batch는 list, 256
            # sample은 tuple, 5
            for idx, sample in enumerate(batch):
                is_int = 0
                #if is_print:
                #    print(f"idx : {idx}")
                if isinstance(sample, int):
                    is_int = 1
                     
                obs, action, reward, done, new_obs = batch[idx - is_int]
                # numpy array 형태이고 각 원소는 np.array(list[float]) 타입
                obs_arr.append(obs.cpu().numpy())
                action_arr.append(action.cpu().numpy())
                reward_arr.append(reward.cpu().numpy())
                done_array.append(done.cpu().numpy())
                new_obs_array.append(new_obs.cpu().numpy())

            # array의 각 원소는 torch.
            # obs, new_obs는 내부가 리스트인 torch이고
            # 나머지는 내부가 스칼라인 torch
            obs_arr = np.array(obs_arr)
            action_arr = np.array(action_arr)
            reward_arr = np.array(reward_arr)
            done_array = np.array(done_array)
            new_obs_array = np.array(new_obs_array)

            # convert to tensors
            obs = T.tensor(obs_arr, dtype=T.float).to(self.q_target_net.device)
            actions = T.tensor(action_arr, dtype=T.float).to(self.q_target_net.device)
            rewards = T.tensor(reward_arr, dtype=T.float).to(self.q_target_net.device)
            dones = T.tensor(done_array, dtype=T.float).to(self.q_target_net.device)
            new_obs = T.tensor(new_obs_array, dtype=T.float).to(self.q_target_net.device)

            # no update on the target net -> use no_grad
            with T.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_target_net(new_obs)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.unsqueeze(1) # 256에서 256 * 1로 변환
                # 1-dones -> reward + 0 if step is last in episode
                # shape : target_q_values - 256 * 256 (비정상, 256 * 1이 정상), next_q_values - 256 (비정상, 256 * 1이 정상)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # get all current Q-values for each obs
            # shape : 256 * 10 (정상)
            current_q_values = self.q_net(obs)


            # choose Q-Values according to actions
            # shape : 256 * 1 (정상)
            current_q_values = T.gather(current_q_values, dim=1, index=actions.long())


            # 추가
            errors = T.abs(current_q_values - target_q_values).data.numpy()

            # priority 업데이트
            for i in range(self.batch_size):
                idx = idxs[i]
                self.memory_buffer.update(idx, errors[i])

            # loss computation. MSE also possible
            # 기존
            # loss = F.smooth_l1_loss(current_q_values, target_q_values)
            # 수정
            loss = (T.FloatTensor(is_weights) * F.mse_loss(current_q_values, target_q_values)).mean()
            losses.append(loss.item())
            # update
            self.q_net.optimizer.zero_grad()
            loss.backward()
            # clip
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_net.optimizer.step()

        self.n_updates += self.gradient_steps

        loss_mean = sum(losses) / len(losses)

        # logs
        self.logger.record(
            {
                'agent_training/exploration rate': self.epsilon,
                'agent_training/n_updates': self.n_updates,
                # 'agent_training/loss': np.mean(losses),
                'agent_training/loss': loss_mean,
                'agent_training/mean_rwd': np.mean(self.reward_info)
            }
        )
        self.logger.dump()

        # if self.num_timesteps % 10_000 == 0:
        #     print(f'Update at {self.num_timesteps} Mean reward {np.mean(self.reward_info)}')

    def on_step(self, total_timesteps):
        """
        Method track and check plenty conditions to e.g. check if q_target_net or epsilon update are necessary
        """
        # update progress
        self.remaining_progress = 1 - float(self.num_timesteps) / float(total_timesteps)

        # update target_net with parameters from main q_net
        if self.num_timesteps % self.target_net_update == 0:
            self.q_target_net.load_state_dict(self.q_net.state_dict())

        # update epsilon
        if (1-self.remaining_progress) > self.fraction_eps:
            # constant if fraction reached
            self.epsilon = self.final_eps
        else:
            # linear function. Goes from initial eps to final eps. Reaches final values in the step,
            # where the function turns constant
            self.epsilon = self.initial_eps + \
                           (1-self.remaining_progress) * (self.final_eps-self.initial_eps) / self.fraction_eps

    # 추가
    def append_sample(self, state, action, reward, done, next_state):
        
        state = T.tensor([state], dtype=T.float).squeeze().to(self.q_target_net.device)
        action = T.tensor([action], dtype=T.float).to(self.q_target_net.device)
        reward = T.tensor([reward], dtype=T.float).to(self.q_target_net.device)
        done = T.tensor([done], dtype=T.float).to(self.q_target_net.device)
        next_state = T.tensor([next_state], dtype=T.float).squeeze().to(self.q_target_net.device)

        # no update on the target net -> use no_grad
        with T.no_grad():
            # Compute the next Q-values using the target network
            next_q_values = self.q_target_net(next_state)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim = 0)
            # 1-dones -> reward + 0 if step is last in episode
            target_q_values = reward + (1 - done) * self.gamma * next_q_values

        # get all current Q-values for each obs
        current_q_values = self.q_net(state)

        # choose Q-Values according to actions
        current_q_values = T.gather(current_q_values, dim=0, index=action.long())

        error = abs(current_q_values[0].item() - target_q_values[0].item())

        self.memory_buffer.store_memory(error, state, action, reward, done, next_state)

    def learn(self, total_instances: int, total_timesteps: int, intermediate_test=None) -> None:
        """
        Learn over n problem instances or n timesteps (environment steps).
        Breaks depending on which condition is met first.
        One learning iteration consists of collecting rollouts and training the networks on the rollout data

        :param total_instances: Instance limit
        :param total_timesteps: Timestep limit
        :param intermediate_test: (IntermediateTest) intermediate test object. Must be created before.

        """
        instances = 0

        # iterate over n episodes = the agents has n episodes to interact with the environment
        for _ in range(total_instances):
            obs = self.env.reset()
            done = False
            instances += 1
            episode_reward = 0

            # run agent on env until done
            while not done:
                # observe and fill buffer
                action = self.get_action(obs)
                new_obs, reward, done, info = self.env.step(action)
                self.num_timesteps += 1
                episode_reward += reward
                # error 추가
                self.append_sample(obs, action, reward, done, new_obs)

                # call intermediate_test on_step
                if intermediate_test:
                    intermediate_test.on_step(self.num_timesteps, instances, self)

                # call function intern on_step
                self.on_step(total_timesteps)

                # break learn if total_timesteps are reached
                if self.num_timesteps >= total_timesteps:
                    print(f'Total timesteps reached: {total_timesteps}')
                    self.logger.record(
                        {
                            'results_on_train_dataset/instances': instances,
                            'results_on_train_dataset/num_timesteps': self.num_timesteps
                        }
                    )
                    self.logger.dump()

                    return None

                # train if training_starts is reached and then every n rollout_steps
                if self.num_timesteps >= self.training_starts and self.num_timesteps % self.train_freq == 0:

                    self.train()
                    # switch back to eval mode
                    self.q_net.train(False)

                obs = new_obs

            self.reward_info.append(episode_reward)

            if instances % len(self.env.data) == len(self.env.data) - 1:
                mean_training_reward = np.mean(self.env.episodes_rewards)
                mean_training_makespan = np.mean(self.env.episodes_makespans)
                mean_training_tardiness = np.mean(self.env.tardiness)
                self.logger.record(
                    {
                        'results_on_train_dataset/mean_reward': mean_training_reward,
                        'results_on_train_dataset/mean_makespan': mean_training_makespan,
                        'results_on_train_dataset/mean_tardiness': mean_training_tardiness
                    }
                )
                self.logger.dump()

        print("TRAINING DONE")
        self.logger.record(
            {
                'results_on_train_dataset/instances': instances,
                'results_on_train_dataset/num_timesteps': self.num_timesteps
            }
        )
        self.logger.dump()
