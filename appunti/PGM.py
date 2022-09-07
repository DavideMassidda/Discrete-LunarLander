import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import pdb

class PolicyCP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(4, 150)
        #self.dense2 = nn.Linear(128, 32)
        self.output = nn.Linear(150, 2)
        self.probab = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))
        #x = F.relu(self.dense2(x))
        x = self.output(x)
        x = self.probab(x)
        return x

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(8, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 4)
        self.probab = nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.output(x)
        x = self.probab(x)
        return x

class Autopilot:
    def __init__(self, environment, model=Policy, gamma=0.99, memory_buffer=2**16, replay_buffer=2**6, optimizer=torch.optim.Adam, **kwargs):
        # Environment features
        self.n_actions = environment.action_space.n
        self.action_space = np.array([*range(environment.action_space.n)])
        self.gamma = gamma
        # Experience replay features
        if replay_buffer > memory_buffer:
            replay_buffer = memory_buffer
        self.memory_buffer = memory_buffer
        self.replay_buffer = replay_buffer
        self.replay_size = 0
        self.memory_stack = deque(maxlen=memory_buffer)
        # Initialize the policy model
        self.model = model()
        self.model.eval()
        self.optimizer = optimizer(params=self.model.parameters(), **kwargs)

    def __get_action(self, act_prob):
        act_prob = act_prob.data.numpy()
        # Replace NaN, is any
        for j in range(self.n_actions):
            if np.isnan(act_prob[j]):
                act_prob = np.full(self.n_actions, fill_value=0.5/self.n_actions)
                act_prob /= np.sum(act_prob)
                break
        # Choice sampling with the policy
        action = np.random.choice(self.action_space, p=act_prob)
        return action

    def decide(self, state):
        with torch.no_grad():
            act_prob = self.model(torch.from_numpy(state).float())
        return self.__get_action(act_prob)

    def __decide(self, state):
        act_prob = self.model(torch.from_numpy(state).float())
        return self.__get_action(act_prob)

    def get_returns_old(self, rewards):
        nt = len(rewards)
        gamma_exp = self.gamma ** np.flip(np.arange(nt))
        disc_rewards = np.zeros(nt)
        for t in range(nt):
            disc_rewards[t] = np.sum(rewards[t:] * gamma_exp[t:])
        disc_rewards /= disc_rewards.max()
        return disc_rewards

    def get_returns(self, rewards):
        times = torch.arange(len(rewards)).float().flip(dims=(0,))
        disc_rewards = torch.pow(self.gamma, times) * rewards#.flip(dims=(0,))
        disc_rewards /= disc_rewards.max()
        return disc_rewards

    def get_loss(self, preds, disc_rewards):
        return -1 * torch.sum(disc_rewards * torch.log(preds)) # e non torch.sum

    def __learn(self, transitions):
        # Set model in train mode
        self.model.train()
        # Convert lists to arrays
        state_batch = torch.tensor(np.float32(transitions[0])) # state
        action_batch = torch.tensor(np.int64(transitions[1])) # action
        reward_batch = torch.tensor(np.float32(transitions[2])) # reward
        # Calculate total rewards
        return_batch = self.get_returns(reward_batch)
        # Recomputes the action-probabilities for all the states in the episode
        pred_batch = self.model(state_batch)
        # Select the predicted probabilities of the actions that were actually taken
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()
        # Adjust model weights
        loss = self.get_loss(prob_batch, return_batch)
        # Zero the gradient
        self.optimizer.zero_grad()
        # Backward propagate the loss
        loss.backward()
        # Adjust model parameters
        self.optimizer.step()
        # Restore the evaluation mode for the model
        self.model.eval()

    def train(self, environment, n_episodes):
        print('# start training')
        self.model.train()
        total_rewards = []  # Store rewards obtained during episodes for output
        for i in range(n_episodes):
            if i % 10 == 0:
                print('# iter:', i)
            total_rewards.append(0.0) # Initialize the episode reward
            curr_state = environment.reset() # Reset the environment and obtain the initial state
            done = False # Will be True when the episode is end
            transitions = [[], [], []]
            while not done:
                # Take an action
                action = self.__decide(curr_state)
                prev_state = curr_state
                curr_state, reward, done, info = environment.step(action)
                # Store the episode into memory
                self.memory_stack.append((prev_state, action, reward))
                # Save the history of the episode
                transitions[0].append(prev_state)
                transitions[1].append(action)
                transitions[2].append(reward)
                # Trace the performance of the episode
                total_rewards[i] += reward
            # Linearly increase the replay batch size (replay_size) until the saturation (replay_buffer)
            if self.replay_size < self.replay_buffer:
                self.replay_size += 1
            # Evaluate the performance and learn from experience
            if self.replay_size > 1:
                batch = self.replay_experience()
                # Add experience to transitions
                transitions[0] = np.vstack([transitions[0], batch[0]])
                transitions[1] = transitions[1] + batch[1]
                transitions[2] = transitions[2] + batch[2]
            # Learn from the episode
            self.__learn(transitions)
        self.model.eval()
        print('# end training')
        return total_rewards
