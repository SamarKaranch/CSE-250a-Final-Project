# ---------------------------
# Install dependencies if needed:
# ---------------------------
# pip install torch torchvision
# pip install gym==0.25.2
# pip install gym-super-mario-bros==7.4.0
# pip uninstall nes-py
# pip install git+https://github.com/Kautenja/nes-py.git
# pip install matplotlib
# pip install tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from collections import deque
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------
# Environment setup
# ---------------------------
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, [["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"]])
n_actions = env.action_space.n


# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(frame):
    frame = Image.fromarray(frame).convert("L").resize((84, 84))
    return np.array(frame, dtype=np.float32) / 255.0


# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------
# DQN Model
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Hyperparameters
# ---------------------------
gamma = 0.99
lr = 1e-4
buffer_capacity = 10000
batch_size = 32
sync_target_steps = 1000
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
online_net = DQN((1, 84, 84), n_actions).to(device)
target_net = DQN((1, 84, 84), n_actions).to(device)
target_net.load_state_dict(online_net.state_dict())
optimizer = optim.Adam(online_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_capacity)


# ---------------------------
# Epsilon-greedy
# ---------------------------
def epsilon_by_frame(frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
        -1.0 * frame_idx / epsilon_decay
    )


# ---------------------------
# Stats tracking
# ---------------------------
episode_rewards = []
episode_lengths = []
episode_losses = []
episode_mean_q = []

# ---------------------------
# Training loop (episode-based)
# ---------------------------
num_episodes = 50  # total episodes to train
frame_idx = 0

pbar = tqdm(total=num_episodes, desc="Training Episodes")

for episode in range(num_episodes):
    state = env.reset()
    state = np.expand_dims(preprocess(state), axis=0)

    done = False
    total_reward = 0
    total_loss = 0
    total_q = 0
    episode_steps = 0

    while not done:
        epsilon = epsilon_by_frame(frame_idx)
        frame_idx += 1
        episode_steps += 1

        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            )
            q_values = online_net(state_tensor)
            action = q_values.argmax(1).item()
            total_q += q_values.max().item()

        # Step in environment
        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(preprocess(next_state), axis=0)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # Learning
        loss_val = 0
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(
                batch_size
            )
            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            with torch.no_grad():
                next_Q = target_net(next_states).max(1)[0]
                td_target = rewards + gamma * next_Q * (1 - dones)

            q_values_batch = (
                online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            )
            loss = F.mse_loss(q_values_batch, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

        # Sync target network
        if frame_idx % sync_target_steps == 0:
            target_net.load_state_dict(online_net.state_dict())

    # End of episode stats
    mean_q = total_q / episode_steps if episode_steps > 0 else 0
    mean_loss = total_loss / episode_steps if episode_steps > 0 else 0

    episode_rewards.append(total_reward)
    episode_lengths.append(episode_steps)
    episode_losses.append(mean_loss)
    episode_mean_q.append(mean_q)

    pbar.set_postfix(
        {
            "Reward": f"{total_reward:.2f}",
            "Steps": episode_steps,
            "Mean Q": f"{mean_q:.2f}",
            "Mean Loss": f"{mean_loss:.4f}",
        }
    )
    pbar.update(1)

pbar.close()

# ---------------------------
# Plotting at the end
# ---------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(episode_mean_q)
plt.title("Mean Q Value vs Episode")
plt.xlabel("Episode")
plt.ylabel("Mean Q Value")

plt.subplot(2, 2, 2)
plt.plot(episode_rewards)
plt.title("Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(2, 2, 3)
plt.plot(episode_losses)
plt.title("Mean Loss vs Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.subplot(2, 2, 4)
plt.plot(episode_lengths)
plt.title("Episode Length vs Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")

plt.tight_layout()
plt.show()
