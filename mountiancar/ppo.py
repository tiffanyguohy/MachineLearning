from sys import stdout
from time import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np


class PI_Network(nn.Module):

    def __init__(self, obs_dim, action_dim, lower_bound, upper_bound):

        super().__init__()
        (
            self.lower_bound,
            self.upper_bound
        ) = (
            torch.tensor(lower_bound, dtype=torch.float32),
            torch.tensor(upper_bound, dtype=torch.float32)
        )
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):

        y = F.tanh(self.fc1(obs))
        y = F.tanh(self.fc2(y))
        action = self.fc3(y)

        action = ((action + 1) * (self.upper_bound - self.lower_bound) / 2 +
                  self.lower_bound)

        return action


def compute_return_advantage(
        rewards, values, is_last_terminal, gamma, gae_lambda, last_value):
    '''
    Computes returns and advantage based on generalized advantage estimation.
    '''
    N = rewards.shape[0]
    advantages = np.zeros(
        (N, 1),
        dtype=np.float32
    )

    tmp = 0.0
    for k in reversed(range(N)):
        if k == N - 1:
            next_non_terminal = 1 - is_last_terminal
            next_values = last_value
        else:
            next_non_terminal = 1
            next_values = values[k+1]

        delta = (rewards[k] +
                 gamma * next_non_terminal * next_values -
                 values[k])

        tmp = delta + gamma * gae_lambda * next_non_terminal * tmp

        advantages[k] = tmp

    returns = advantages + values

    return returns, advantages


class PPOBuffer:
    '''
    Implementation of PPO buffer

    Author:     Niranjan Bhujel
    Date:       Aug 3, 2022
    '''

    def __init__(self, obs_dim, action_dim, buffer_capacity, seed=None):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_capacity = buffer_capacity

        self.obs = np.zeros(
            shape=(self.buffer_capacity, self.obs_dim),
            dtype=np.float32
        )
        self.action = np.zeros(
            shape=(self.buffer_capacity, self.action_dim),
            dtype=np.float32
        )
        self.reward = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.log_prob = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.returns = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.advantage = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.values = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )

        self.rng = np.random.default_rng(seed=seed)
        self.start_index, self.pointer = 0, 0

    def record(self, obs, action, reward, values, log_prob):

        self.obs[self.pointer] = obs
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.values[self.pointer] = values
        self.log_prob[self.pointer] = log_prob

        self.pointer += 1

    def process_trajectory(self, gamma, gae_lam, is_last_terminal, last_v):
        path_slice = slice(self.start_index, self.pointer)
        values_t = self.values[path_slice]

        self.returns[path_slice], self.advantage[path_slice] = (
                compute_return_advantage(self.reward[path_slice], values_t,
                                         is_last_terminal, gamma, gae_lam,
                                         last_v))

        self.start_index = self.pointer

    def get_data(self):
        whole_slice = slice(0, self.pointer)
        return {
            'obs': self.obs[whole_slice],
            'action': self.action[whole_slice],
            'reward': self.reward[whole_slice],
            'values': self.values[whole_slice],
            'log_prob': self.log_prob[whole_slice],
            'return': self.returns[whole_slice],
            'advantage': self.advantage[whole_slice],
        }

    def get_mini_batch(self, batch_size):
        assert batch_size <= self.pointer, \
                'Batch size must be smaller than number of data.'
        indices = np.arange(self.pointer)
        self.rng.shuffle(indices)
        split_indices = []
        point = batch_size
        while point < self.pointer:
            split_indices.append(point)
            point += batch_size

        temp_data = {
            'obs': np.split(self.obs[indices], split_indices),
            'action': np.split(self.action[indices], split_indices),
            'reward': np.split(self.reward[indices], split_indices),
            'values': np.split(self.values[indices], split_indices),
            'log_prob': np.split(self.log_prob[indices], split_indices),
            'return': np.split(self.returns[indices], split_indices),
            'advantage': np.split(self.advantage[indices], split_indices),
        }

        n = len(temp_data['obs'])
        data_out = []
        for k in range(n):
            data_out.append(
                {
                    'obs': temp_data['obs'][k],
                    'action': temp_data['action'][k],
                    'reward': temp_data['reward'][k],
                    'values': temp_data['values'][k],
                    'log_prob': temp_data['log_prob'][k],
                    'return': temp_data['return'][k],
                    'advantage': temp_data['advantage'][k],
                }
            )

        return data_out

    def clear(self):
        self.start_index, self.pointer = 0, 0


class PPOPolicy(nn.Module):

    def __init__(self, pi_network, v_network, learning_rate, clip_range,
                 value_coeff, obs_dim, action_dim, initial_std=1.0,
                 max_grad_norm=0.5):

        super().__init__()

        (
            self.pi_network,
            self.v_network,
            self.learning_rate,
            self.clip_range,
            self.value_coeff,
            self.obs_dim,
            self.action_dim,
            self.max_grad_norm,
        ) = (
            pi_network,
            v_network,
            learning_rate,
            clip_range,
            value_coeff,
            obs_dim,
            action_dim,
            max_grad_norm
        )

        # Gaussian policy will be used. So, log standard deviation is created
        # as trainable variables
        self.log_std = nn.Parameter(torch.ones(self.action_dim) *
                                    torch.log(torch.tensor(initial_std)),
                                    requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

    def forward(self, obs, report=False):

        pi_out = self.pi_network(obs)

        # Add Normal distribution layer at the output of pi_network
        dist_out = Normal(pi_out, torch.exp(self.log_std))

        v_out = self.v_network(obs)

        if report:
            print('forward:    observation = ', obs)
            print('forward:    pi_out = ', pi_out)
            print('forward:    dist_out = ', dist_out)
            print('forward:    v_out = ', v_out)

        return dist_out, v_out

    def get_action(self, obs, report=False):
        '''
        Sample action based on current policy
        '''

        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)

        dist, values = self.forward(obs_torch, report)

        action = dist.sample()

        log_prob = torch.sum(dist.log_prob(action), dim=1)

        if report:
            print('get_action: action = ', action[0].detach().numpy())
            print('get_action: log_prob = ',
                  torch.squeeze(log_prob).detach().numpy())

        return (action[0].detach().numpy(),
                torch.squeeze(log_prob).detach().numpy(),
                torch.squeeze(values).detach().numpy())

    def get_values(self, obs):
        '''
        Function  to return value of the state
        '''
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)

        _, values = self.forward(obs_torch)

        return torch.squeeze(values).detach().numpy()

    def evaluate_action(self, obs_batch, action_batch, training):
        '''
        Evaluate taken action.
        '''
        obs_torch = obs_batch.clone().detach()
        action_torch = action_batch.clone().detach()
        dist, values = self.forward(obs_torch)
        log_prob = dist.log_prob(action_torch)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return log_prob, values

    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch,
               return_batch):
        '''
        Performs one step gradient update of policy and value network.
        '''

        new_log_prob, values = self.evaluate_action(
                obs_batch, action_batch, training=True)

        ratio = torch.exp(new_log_prob-log_prob_batch)
        clipped_ratio = torch.clip(
            ratio,
            1-self.clip_range,
            1+self.clip_range,
        )

        surr1 = ratio * advantage_batch
        surr2 = clipped_ratio * advantage_batch
        pi_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = self.value_coeff * torch.mean((values - return_batch)**2)
        total_loss = pi_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            pi_loss.detach(),
            value_loss.detach(),
            total_loss.detach(),
            (torch.mean((ratio - 1) - torch.log(ratio))).detach(),
            torch.exp(self.log_std).detach()
        )


class V_Network(nn.Module):

    def __init__(self, obs_dim) -> None:
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        y = F.tanh(self.fc1(obs))
        y = F.tanh(self.fc2(y))
        values = self.fc3(y)

        return values

def _report_training_progress(
        timestep, total_timesteps, start_time, mean_ep_reward):

    if mean_ep_reward is not None:
        print('%7d / %d steps | mean reward = %f | %d steps / sec' % (
              timestep, total_timesteps, mean_ep_reward,
              int((timestep / (time() - start_time)))))



def train(env,
          save=None,
          num_steps=2048,
          batch_size=64,
          reps_per_step=500,
          gamma=0.99,
          gae_lam=0.95,
          num_epochs=10,
          report_steps=1000):
    '''
    Returns array of mean rewards
    '''

    print('Training ...')
    stdout.flush()

    total_timesteps = num_steps * reps_per_step

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    pi_network = PI_Network(
            obs_dim, action_dim, lower_bound, upper_bound)

    v_network = V_Network(obs_dim)

    learning_rate = 3e-4

    buffer = PPOBuffer(obs_dim, action_dim, num_steps)

    policy = PPOPolicy(
        pi_network,
        v_network,
        learning_rate,
        clip_range=0.2,
        value_coeff=0.5,
        obs_dim=obs_dim,
        action_dim=action_dim,
        initial_std=1.0,
        max_grad_norm=0.5,
    )

    ep_reward = 0.0
    ep_count = 0

    pi_losses, v_losses, total_losses, approx_kls, stds = [[]] * 5
    mean_rewards = []

    obs, _ = env.reset()

    start_time = time()

    mean_ep_reward = None

    for timestep in range(total_timesteps):

        try:

            action, log_prob, values = policy.get_action(obs)

            clipped_action = np.clip(action, lower_bound, upper_bound)

            next_obs, reward, terminated, truncated, _ = (
                    env.step(clipped_action))

            done = terminated or truncated

            ep_reward += reward

            # Add to buffer
            buffer.record(obs, action, reward, values, log_prob)

            obs = next_obs

            # Calculate advantage and returns if it is the end of episode
            # or its time to update
            if done or (timestep + 1) % num_steps == 0:
                if done:
                    ep_count += 1
                # Value of last time-step
                last_value = policy.get_values(obs)

                # Compute returns and advantage and store in buffer
                buffer.process_trajectory(
                    gamma=gamma,
                    gae_lam=gae_lam,
                    is_last_terminal=done,
                    last_v=last_value)
                obs, _ = env.reset()

            if (timestep + 1) % num_steps == 0:
                # Update for epochs
                for ep in range(num_epochs):
                    batch_data = buffer.get_mini_batch(batch_size)
                    num_grads = len(batch_data)

                    # Iterate over minibatch of data
                    for k in range(num_grads):
                        (
                            obs_batch,
                            action_batch,
                            log_prob_batch,
                            advantage_batch,
                            return_batch,
                        ) = (
                            batch_data[k]['obs'],
                            batch_data[k]['action'],
                            batch_data[k]['log_prob'],
                            batch_data[k]['advantage'],
                            batch_data[k]['return'],
                        )

                        # Normalize advantage
                        advantage_batch = (
                            advantage_batch -
                            np.squeeze(np.mean(advantage_batch, axis=0))
                        ) / (np.squeeze(np.std(advantage_batch, axis=0))
                             + 1e-8)

                        # Convert to torch tensor
                        (
                            obs_batch,
                            action_batch,
                            log_prob_batch,
                            advantage_batch,
                            return_batch,
                        ) = (
                            torch.tensor(obs_batch, dtype=torch.float32),
                            torch.tensor(action_batch,
                                         dtype=torch.float32),
                            torch.tensor(log_prob_batch,
                                         dtype=torch.float32),
                            torch.tensor(advantage_batch,
                                         dtype=torch.float32),
                            torch.tensor(return_batch,
                                         dtype=torch.float32),
                        )

                        # Update the networks on minibatch of data
                        (
                            pi_loss,
                            v_loss,
                            total_loss,
                            approx_kl,
                            std,
                        ) = policy.update(obs_batch, action_batch,
                                          log_prob_batch, advantage_batch,
                                          return_batch)

                        pi_losses.append(pi_loss.numpy())
                        v_losses.append(v_loss.numpy())
                        total_losses.append(total_loss.numpy())
                        approx_kls.append(approx_kl.numpy())
                        stds.append(std.numpy())

                buffer.clear()

                mean_ep_reward = (ep_reward / ep_count
                                  if ep_count > 0 else 0)

                ep_reward, ep_count = 0.0, 0

                mean_rewards.append(mean_ep_reward)
                pi_losses, v_losses, total_losses, approx_kls, stds = (
                        [], [], [], [], [])

            if timestep > 0 and timestep % report_steps == 0:
                _report_training_progress(timestep, total_timesteps,
                                               start_time, mean_ep_reward)

        except KeyboardInterrupt:

            print('\nTraining interrupted')
            return None, None

    _report_training_progress(
            timestep+1, total_timesteps, start_time, mean_ep_reward)

    print('Done training')

    # Save policy network if indicated
    if save is not None:
        print('Saving ' + save)
        torch.save(pi_network.state_dict(), save)

    return mean_rewards

def test(env, filename):

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    pi_network = PI_Network(obs_dim, action_dim, lower_bound, upper_bound)

    pi_network.load_state_dict(torch.load(filename))

    obs, _ = env.reset()

    while True:

        try:

            obs_torch = torch.unsqueeze(
                    torch.tensor(obs, dtype=torch.float32), 0)

            action = pi_network(obs_torch).detach().numpy()

            clipped_action = np.clip(
                    action[0], lower_bound, upper_bound)

            obs, reward, terminated, truncated, _ = (
                    env.step(clipped_action))

            if terminated or truncated:
                break

        except KeyboardInterrupt:

            break

    env.close()
