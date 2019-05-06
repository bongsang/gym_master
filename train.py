"""
 - Author: Bongsang Kim
 - Contact: happykbs@gmail.com
 - GitHub: https://github.com/bongsang/
 - LinkedIn: https://www.linkedin.com/in/bongsang/
"""

import numpy as np
from collections import deque

from bongsang.dqn.dqn_core import DQN
from gym.envs import registration as gym


def train(env_name, device, episodes=100000, epochs=1000):
    # Create environment
    env = gym.make(env_name)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create Deep RL agent
    agent = DQN(device, state_size=observation_size, action_size=action_size, seed=0)

    scores = deque(maxlen=100)
    success_score = 195
    for episode in range(episodes):
        state = env.reset()
        env.render()
        state = np.reshape(state, [1, observation_size])
        epoch_reward = 0
        for epoch in range(epochs):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            epoch_reward += reward
            print(f'Episode={episode} \t Epoch={epoch} \t Action={action} \t Reward={reward} \t Next State={next_state}')

            # print(f'next state size = {np.shape(next_state)}')
            next_state = np.reshape(next_state, [1, observation_size])
            agent.step(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        scores.append(epoch_reward)
        mean_score = np.mean(scores)

        if mean_score >= success_score and episode >= 100:
            print(f'Episode = {episode}, Solved after {episode-100} trials.')


if __name__ == '__main__':
    train(env_name='CartPole-v1', device='gpu')

