"""
 - Author: Bongsang Kim
 - Contact: happykbs@gmail.com
 - GitHub: https://github.com/bongsang/
 - LinkedIn: https://www.linkedin.com/in/bongsang/
"""

import numpy as np

from bongsang.dqn.dqn_core import DQN
from gym.envs import registration as gym


# import matplotlib.pyplot as plt
# from collections import deque
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import csv
# from datetime import datetime
#
# import os
# from time import time, sleep


def train(env_name, device, episodes=100000, epochs=1000):
    # Create environment
    env = gym.make(env_name)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create Deep RL agent
    agent = DQN(device, state_size=4, action_size=5, seed=0)

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, observation_size])
        for epoch in range(epochs):
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            reward = reward if not done else -reward
            next_state = np.reshape(next_state, [1, observation_size])

            agent.step(state[0], action, reward, next_state[0], done)

            state = next_state
            if done:
                break


if __name__ == '__main__':
    train(env_name='CartPole-v1', device='gpu')



#
#
#
#     # env = env(mode='heating', sp=sp, pv_init=pv_init, loop=loop)
#     env = LG_env(mode='hotwater', sp=sp, pv_init=pv_init, loop=loop)
#     env.make('EnergyBalance')
#     observation_size = env.observation_size
#     action_size = env.action_size
#
#     agent = DQN(device, state_size=4, action_size=5, seed=0)
#
#     n_episodes=episodes
#
#     max_t=1000
#     eps_start=1.0
#     # eps_end=0.01
#     eps_end=0.01
#     eps_decay=0.99993
#
#     rewards = []                        # list containing rewards from each episode
#     rewards_window = deque(maxlen=10000)  # last 100 rewards
#     eps = eps_start                    # initialize epsilon
#
#     Kp_list = []
#     Ki_list = []
#     reward = -100
#     previous_reward = -100
#     state, _, _ = env.step(0)  # For EnergyBalance
#     for i_episode in range(1, n_episodes+1):
#         action = agent.act(state, eps)
#         next_state, reward, info = env.step(action)  # For EnergyBalance
#         done = True
#         agent.step(state[0], action, reward, next_state[0], done)
#         state = next_state
#         eps = max(eps_end, eps_decay*eps)   # decrease epsilon
#         if len(Kp_list) > 0:
#             print('\rEpisode={}\tState={}\tAction={}\tReward={}\tKp={}\tKi={}\tEpsilon={}'.format(i_episode, state[0], action, reward, Kp_list[-1], Ki_list[-1],eps  ))
#
#         # Selecting only better reward than the previous one.
#         # if reward >= previous_reward:
#         #     previous_reward = reward
#         #     rewards_window.append(reward)       # save most recent reward
#         #     rewards.append(reward)              # save most recent reward
#         #     Kp_list.append(info['Kp'])  # For EnergyBalance
#         #     Ki_list.append(info['Ki'])  # For EnergyBalance
#
#
#         previous_reward = reward
#         rewards_window.append(reward)       # save most recent reward
#         rewards.append(reward)              # save most recent reward
#         Kp_list.append(info['Kp'])  # For EnergyBalance
#         Ki_list.append(info['Ki'])  # For EnergyBalance
#
#     from datetime import datetime
#     timestamp = datetime.now().strftime("%Y%m%d%H%M")
#
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     # local_model_dir = dir_path + '/models/dqn_local_model_{}.pt'.format(timestamp)
#     # target_model_dir = dir_path + '/models/dqn_target_model_{}.pt'.format(timestamp)
#     local_model_dir = dir_path + '/models/dqn_local_model.pt'
#     target_model_dir = dir_path + '/models/dqn_target_model.pt'
#
#     torch.save(agent.qnetwork_local.state_dict(), local_model_dir)
#     torch.save(agent.qnetwork_target.state_dict(), target_model_dir)
#     print('Successfully model saved! ==> {}!'.format(local_model_dir))
#     print('Successfully model saved! ==> {}!'.format(target_model_dir))
#
#
#     # Visualization
#     f, axes = plt.subplots(3, sharex=True)
#     axes[0].plot(np.arange(len(rewards)), rewards)
#     axes[0].set_title('Reward = {}'.format(rewards[-1]))
#
#     axes[1].plot(np.arange(len(Kp_list)), Kp_list)
#     axes[1].set_title('Kp = {}'.format(Kp_list[-1]))
#
#     axes[2].plot(np.arange(len(Ki_list)), Ki_list)
#     axes[2].set_title('Ki = {}'.format(Ki_list[-1]))
#
#     with open('./logs/pid.csv', mode='w') as f:
#         w = csv.writer(f, delimiter=',')
#         w.writerow([Kp_list[-1], Ki_list[-1]])
#
#     fig_dir = dir_path + '/models/dqn_model_visualization_{}.png'.format(timestamp)
#     plt.savefig(fig_dir)
#
#
# def inference(device, model_date):
#     env = LG_env(mode='hotwater', sp=sp, pv_init=pv_init, loop=loop)
#     # env.make('EnergyBalance')
#     env.make('LgYangjaeCampus')
#
#     agent = DQN(device, state_size=4, action_size=5, seed=0)
#
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     local_model_dir = dir_path + '/models/dqn_local_model.pt'
#     target_model_dir = dir_path + '/models/dqn_target_model.pt'
#
#     agent.qnetwork_local.load_state_dict(torch.load(local_model_dir))
#     agent.qnetwork_target.load_state_dict(torch.load(target_model_dir))
#     print('Successfully model loaded! ==> {}!'.format(local_model_dir))
#     print('Successfully model loaded! ==> {}!'.format(target_model_dir))
#
#     n_episodes=1
#     max_t=1000
#     eps_start=1.0
#     eps_end=0.01
#     eps_decay=0.99993
#
#     rewards = []                        # list containing rewards from each episode
#     rewards_window = deque(maxlen=10000)  # last 100 rewards
#     eps = eps_start                    # initialize epsilon
#
#
#     Kp_list = []
#     Ki_list = []
#     reward = -100
#     previous_reward = -100
#     f = open('./logs/' + datetime.now().strftime("%Y-%m-%d_XI") + '.csv', 'w')
#     header = ['date','set','supply','valve', 'Kp', 'Ki']
#     writer = csv.DictWriter(f, fieldnames=header)
#     writer.writeheader()
#     f.close()
#     state, _, _ = env.step(0)  # For EnergyBalance
#     while True:
#         action = agent.act(state, eps)
#         next_state, reward, info = env.step(action)  # For EnergyBalance
#         if (datetime.now().hour == 11 and (25 < datetime.now().minute < 30)):
#             break
#         done = True
#         state = next_state
#         eps = max(eps_end, eps_decay*eps)   # decrease epsilon
#         if len(Kp_list) > 0:
#             print('\rEpisode={}\tState={}\tAction={}\tReward={}\tKp={}\tKi={}\tEpsilon={}'.format(n_episodes, state[0], action, reward, Kp_list[-1], Ki_list[-1],eps  ))
#         # for i in range(len(info['pvline'])):
#         #     writer.writerow({'date':info['timeline'][i], 'set': info['baseline'][i], 'supply':round(info['pvline'][i], 2), 'valve':round(info['outline'][i], 2), 'Kp':info['Kp'], 'Ki':info['Ki']})
#
#         if reward >= previous_reward:
#             previous_reward = reward
#             rewards_window.append(reward)       # save most recent reward
#             rewards.append(reward)              # save most recent reward
#             Kp_list.append(info['Kp'])  # For EnergyBalance
#             Ki_list.append(info['Ki'])  # For EnergyBalance
#         n_episodes += 1
#     # f.close()
#
#
#
