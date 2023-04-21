import gymnasium as gym
import LunarLander as ll
import pickle

# Training ----------------------------------------------------

env = gym.make('LunarLander-v2')
agent = ll.Autopilot(env)
agent.train(env, n_episodes=1000)

with open('cache/agent.pickle', 'wb') as file:
    pickle.dump(agent, file)

# Evaluation --------------------------------------------------

with open('cache/agent.pickle', 'rb') as file:
    agent = pickle.load(file)

performances = [ll.play(env, agent) for i in range(1000)]

with open('cache/performances.pickle', 'wb') as file:
    pickle.dump(performances, file)
