import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from copy import deepcopy
import random
import time
import pdb

class Q_Network(nn.Module):
    """
    Feed-forward neural network representing the brain of the autopilot of the space shuttle.
    Given the state of the environment, returns the Q-value for each action.
    """
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(8, 32)
        self.dense2 = nn.Linear(32, 64)
        self.dense3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.output(x)
        return x

class Autopilot:
    """
    Autopilot of the space shuttle (agent)
    """
    def __init__(self, environment, model=Q_Network, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.98, gamma=0.99, memory_buffer=2**16, replay_buffer=64, lr=0.001):
        """
        :param environment:
        :param model:
        :param epsilon_start:
        :param epsilon_min:
        :param epsilon_decay:
        :param gamma:
        :param memory_buffer:
        :param replay_buffer:
        :param optimizer:
        :param kwargs:
        """
        # Environment features
        self.action_space = np.array([*range(environment.action_space.n)])
        # Epsilon-greedy strategy
        self.epsilon_value = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Discount factor for rewards
        self.gamma = gamma
        # Experience replay features
        if replay_buffer > memory_buffer:
            replay_buffer = memory_buffer
        self.memory_buffer = memory_buffer
        self.replay_buffer = replay_buffer
        self.replay_size = 0
        self.memory_stack = deque(maxlen=memory_buffer)
        # Trace the training history
        self.train_rewards = []
        # Initialize the Deep Q-network
        self.model = model() # leading network
        self.target = model() # target network
        self.model.eval()
        self.target.eval()
        # Optimization features
        self.loss_fn = nn.HuberLoss(delta=10) #nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

    def decide(self, state):
        """
        Decide the action without randomness and without computing gradient (production mode)

        :param state: a state vector produced by the environment.
        """
        with torch.no_grad():
            Q_values = self.model(torch.from_numpy(state))
        index = np.argmax(Q_values.data.numpy())
        action = self.action_space[index]
        return action

    def __decide(self, state):
        """
        Decide the action with randomness (training mode)

        :param state: a state vector produced by the environment.
        """
        if (random.random() < self.epsilon_value):
            action = np.random.choice(self.action_space)
        else:
            action = self.decide(state)
        return action

    def __reshape_batch(self, batch):
        """
        Reshape a data batch stacking elements

        :param batch: tuple, a data batch from the training loop.
        """
        prev_state = []
        action = []
        reward = []
        curr_state = []
        done = []
        for (s1,a,r,s2,d) in batch:
            prev_state.append(s1)
            action.append(a)
            reward.append(r)
            curr_state.append(s2)
            done.append(d)
        return (prev_state, action, reward, curr_state, done)

    def __learn(self, batch):
        """
        Calculate Q-values and update model parameters

        :param batch: a tuple containing:
        - state1: numpy array, state before taking an action.
        - action: integer list, action taken at t1.
        - reward: numeric list, reward obtained taking the action.
        - state2: numpy array, state after taking an action.
        - done: boolean list, it specifies if the episode is terminated.
        """
        # Reshape the experience batch stacking elements in vectors
        # batch_i = (prev_state, action, reward, curr_state, done)
        batch = self.__reshape_batch(batch)
        #batch = tuple(map(list, zip(*batch)))
        # Convert inputs to torch tensors
        state1 = torch.from_numpy(np.stack(batch[0], axis=0))
        state2 = torch.from_numpy(np.stack(batch[3], axis=0))
        reward = torch.tensor(batch[2])
        undone = torch.tensor(batch[4]).logical_not().float()
        # Convert action in a column vector suitable to be used to masking the tensor returned by the model
        action = torch.tensor(batch[1]).long().view(-1, 1)
        # Given a state, compute the Q-value for each action and accumulate the gradient
        self.model.train()
        Q1 = self.model(state1)
        self.model.eval()
        # Extract the Q-value corresponding to the action taken
        X = Q1.gather(dim=1, index=action).squeeze(dim=1)
        # From the resulting state, compute the Q-value for each action at t2
        # (t2 is after receiving the reward, so these are the Q-values "in hindsight")
        with torch.no_grad():
            # Values of the next actions according to the online network
            Q2_online = self.model(state2)
            # Values of the next actions according to the target network
            Q2_target = self.target(state2)
        # Action to take according to the online network
        next_action = torch.argmax(Q2_online, dim=1).view(-1,1)
        # Q-value at t2 (corresponding to the action it will is better to take)
        maxQ2 = Q2_target.gather(dim=1, index=next_action).squeeze(dim=1)
        # maxQ2 = torch.max(Q2_target, dim=1)[0] # Vanilla DQN (max Q-value at t2)
        # Correct the maxQ by reinforcing it with reward and apply the discount
        Y = reward + self.gamma * undone * maxQ2
        # Compute the loss
        loss = self.loss_fn(X.float(), Y.float())
        # Zero the gradient
        self.optimizer.zero_grad()
        # Backward propagate the loss
        loss.backward()
        # Adjust model parameters
        self.optimizer.step()

    def train(self, environment, n_episodes):
        """
        Main training loop

        :param environment: the LunarLander-v2 gym environment.
        :param n_episodes: integer, number of episodes to run.
        """
        print('>>> Start training')
        trace_rewards = 0.0
        trace_episodes = 0
        for i in range(n_episodes):
            self.train_rewards.append(0.0) # Initialize the episode reward
            curr_state = environment.reset()[0] # Reset the environment and obtain the initial state
            done = False # Will be True when the episode is end
            while not done: # Run the episode
                # Get the batch from memory
                if self.replay_size > 1:
                    # Sample from the memory stack
                    batch = random.sample(self.memory_stack, k=self.replay_size)
                else:
                    batch = []
                # Choose an action using randomness
                action = self.__decide(curr_state)
                # Take the action
                prev_state = deepcopy(curr_state)
                curr_state, reward, term, trunc = environment.step(action)[:4]
                done = term or trunc
                # Save the transition from state1 to state2
                transition = (prev_state, action, reward, curr_state, done)
                # Append the transition to the training batch
                batch.append(transition)
                # Store the transition into memory
                self.memory_stack.append(transition)
                # Trace the performance of the episode
                self.train_rewards[-1] += reward
                # Evaluate the performance and learn from experience
                self.__learn(batch)
                # Linearly increase the replay batch size until the saturation
                if self.replay_size < self.replay_buffer:
                    self.replay_size += 1
            # Update the target network
            self.target.load_state_dict(self.model.state_dict())
            # Decrease epsilon until the minimum
            self.epsilon_value = max(self.epsilon_value * self.epsilon_decay, self.epsilon_min)
            # Trace the progress
            trace_rewards += self.train_rewards[-1]
            trace_episodes += 1
            if i % 10 == 0:
                trace_rewards /= trace_episodes
                print('# Episode:', i, '\t\tMean reward:', np.round(trace_rewards, 2))
                trace_rewards = 0.0
                trace_episodes = 0
        print('>>> End training')

    def train_history(self, bin_window=0):
        """
        Train history

        :param bin_window: integer, size of the window within which computing the mean.
        :return: numeric list, total reward obtained at each training episode, sorted in time order.
        """
        if bin_window < 2:
            train_rewards = self.train_rewards
        else:
            train_rewards = bin_mean(self.train_rewards, window=bin_window)
        return train_rewards

    def model_save(self, path):
        """
        Export the Q-network checkpoint

        :param path: string, path to file.
        """
        torch.save(self.model.state_dict(), path)

    def model_load(self, path):
        """
        Restore a Q-network checkpoint

        :param path: string, path to file.
        """
        self.model.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.model.state_dict())

def bin_mean(x, window=10):
    """
    Split a time series in bins and returns the mean value for each bin.

    :param x: a time series.
    :param window: integer, size of the window within which computing the mean.
    """
    x_size = len(x)
    n_bins = int(np.ceil(x_size/window))
    one_hot = np.eye(n_bins)
    kernel = []
    for i in range(n_bins):
        kernel.append(np.array([]))
        for j in range(n_bins):
            one_hot_expanded = np.repeat(one_hot[i][j], window)
            kernel[i] = np.append(kernel[i], one_hot_expanded)
    kernel = np.stack(kernel, axis=1)[:x_size,:]
    bin_size = np.sum(kernel, axis=0)
    return x @ kernel / bin_size

def play(agent, sleep=0, random_state=None, render=False):
    """
    Play with the agent: land on the Moon!

    :param agent: an agent of Autopilot class.
    :param sleep: number of seconds elapsing between frames.
    :param random_state: seed for the environment generation.
    :param render: boolean, specifies if render the episode.
    """
    environment = gym.make('LunarLander-v2', render_mode='human')
    if random_state is None:
        state = environment.reset()[0]
    else:
        state = environment.reset(seed=random_state)[0]
    if render == True:
        environment.render()
    else:
        sleep = 0
    done = False
    episode_reward = 0
    duration = 0
    while not done:
        duration += 1
        time.sleep(sleep)
        action = agent.decide(state)
        state, reward, term, trunc = environment.step(action)[:4]
        done = term or trunc
        episode_reward += reward
        if render == True:
            environment.render()
    if render == True:
        time.sleep(sleep * 2)
        environment.close()
    return {'duration': duration, 'reward': episode_reward, 'rest': reward >= 100, 'solved': episode_reward >= 200}

def test(environment, agent, n_episodes):
    """
    Test the agent performance

    :param environment: the LunarLander-v2 gym environment.
    :param agent: an agent of Autopilot class.
    :param n_episodes: integer, number of episodes to run.
    :return: numeric list, total reward obtained at each training episode, sorted in time order.
    """
    total_rewards = [play(environment, agent) for i in range(n_episodes)]
    return total_rewards
