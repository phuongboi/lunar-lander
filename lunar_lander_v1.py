import numpy as np
import pandas as pd
from collections import deque
import random
import gymnasium as gym
import torch
from torch import nn
from matplotlib import pyplot as plt
import pickle

class Network(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
        nn.Linear(num_states, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_actions)
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []

        self.replay_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.num_actions = self.action_space.n
        self.num_states = env.observation_space.shape[0]
        self.model = self.make_model()

    def make_model(self):
        model = Network(self.num_states, self.num_actions)
        return model

    def agent_policy(self, state):
        # epsilon greedy policy
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            q_value = self.model(torch.from_numpy(state))
            action = np.argmax(q_value.detach().numpy())
        return action


    def add_to_replay_buffer(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_from_reply_buffer(self):
        random_sample = random.sample(self.replay_buffer, self.batch_size)
        return random_sample

    def get_memory(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        terminals = np.array([i[4] for i in random_sample])
        return torch.from_numpy(states), torch.from_numpy(actions), rewards, torch.from_numpy(next_states), terminals

    def train_with_relay_buffer(self):
            # replay_memory_buffer size check
        if len(self.replay_buffer) < self.batch_size or self.counter != 0:
            return

        # Early Stopping
        if np.mean(self.rewards_list[-10:]) > 180:
            return

        sample = self.sample_from_reply_buffer()
        states, actions, rewards, next_states, terminals = self.get_memory(sample)

        next_q_mat = self.model(next_states)

        next_q_vec = np.max(next_q_mat.detach().numpy(), axis=1).squeeze()

        target_vec = rewards + self.gamma * next_q_vec* (1 - terminals)
        q_mat = self.model(states)
        q_vec = q_mat.gather(dim=1, index=actions.unsqueeze(1)).type(torch.FloatTensor)
        target_vec = torch.from_numpy(target_vec).unsqueeze(1).type(torch.FloatTensor)
        loss = self.loss_func(q_vec, target_vec)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size


    def train(self, num_episodes=2000, can_stop=True):
        self.model.train()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for episode in range(num_episodes):
            state = env.reset()
            reward_for_episode = 0
            num_steps = 1000
            state = state[0]
            for step in range(num_steps):
                env.render()
                received_action = self.agent_policy(state)
                # print("received_action:", received_action)
                next_state, reward, terminal, info, _ = env.step(received_action)

                # Store the experience in replay memory
                self.add_to_replay_buffer(state, received_action, reward, next_state, terminal)
                # add up rewards
                reward_for_episode += reward
                state = next_state
                self.update_counter()
                self.train_with_relay_buffer()

                if terminal:
                    break
            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each experience completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Check for breaking condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if last_rewards_mean > 200 and can_stop:
                print("DQN Training Complete...")
                break
            print(episode, "\t: Episode || Reward: ",reward_for_episode, "\t|| Average Reward: ",last_rewards_mean, "\t epsilon: ", self.epsilon )

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def test_already_trained_model(trained_model):
    rewards_list = []
    num_test_episode = 100
    env = gym.make("LunarLander-v2")
    print("Starting Testing of the trained model...")

    step_count = 1000

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_states= env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_states])
        reward_for_episode = 0
        for step in range(step_count):
            env.render()
            selected_action = np.argmax(trained_model(current_state)[0])
            new_state, reward, terminal, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_states])
            current_state = new_state
            reward_for_episode += reward
            if terminal:
                break
        rewards_list.append(reward_for_episode)
        print(test_episode, "\t: Episode || Reward: ", reward_for_episode)

    return rewards_list




def plot_df(df, chart_name, title, x_axis_label, y_axis_label):
    plt.rcParams.update({'font.size': 17})
    df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    # plt.ylim((-400, 300))
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)


def plot_df2(df, chart_name, title, x_axis_label, y_axis_label):
    df['mean'] = df[df.columns[0]].mean()
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim((0, 300))
    plt.xlim((0, 100))
    plt.legend().set_visible(False)
    fig = plot.get_figure()
    fig.savefig(chart_name)


def plot_experiments(df, chart_name, title, x_axis_label, y_axis_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    plot = df.plot(linewidth=1, figsize=(15, 8), title=title)
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    plt.ylim(y_limit)
    fig = plot.get_figure()
    fig.savefig(chart_name)

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')

    # env.unwrapped.seed
    # env.seed(env)
    # np.random.seed(21)

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 2000
    print('St')
    model = DQN(env, lr, gamma, epsilon, epsilon_decay)
    model.train(training_episodes, True)

    # Save Everything
    save_dir = "save_weights/"
    # Save trained model
    model.save_model(save_dir + "lunalander_exp1.pt")

    # Save Rewards list
    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

    # plot reward in graph
    reward_df = pd.DataFrame(rewards_list)
    plot_df(reward_df, "Figure 1: Reward for each training episode", "Reward for each training episode", "Episode","Reward")
