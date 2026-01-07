import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from trex_env import DinoEnv  

# ======================================================
# Configuration
# ======================================================
SEED = 42
NUM_EPISODES = 1500
MAX_STEPS = 5000

GAMMA = 0.99
LR = 1e-4

BATCH_SIZE = 64
BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE = 5_000

EPS_START = 0.05     # Low because we start from expert
EPS_END = 0.01
EPS_DECAY = 0.9995

TARGET_UPDATE_FREQ = 1000
TRAIN_FREQ = 4

BC_MODEL_PATH = "BC_0/bc_model_epoch_1000.pth"

LOG_DIR = "DQN_IL_tensorboard"
MODEL_DIR = "DQN_IL_models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

writer = SummaryWriter(LOG_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ======================================================
# Q-Network (SAME architecture as BC)
# ======================================================
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ======================================================
# Replay Buffer
# ======================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# ======================================================
# Epsilon-Greedy Policy
# ======================================================
def select_action(state, q_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_t)
        return torch.argmax(q_values, dim=1).item()


# ======================================================
# Training Step
# ======================================================
def train_step(q_net, target_net, optimizer, replay_buffer):
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    q_values = q_net(states)
    q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(1)[0]
        target_q = rewards + GAMMA * max_next_q * (1 - dones)

    loss = nn.MSELoss()(q_action, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ======================================================
# Main Training Loop
# ======================================================
def main():
    env = DinoEnv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Networks
    q_net = QNetwork(obs_dim, act_dim).to(device)
    target_net = QNetwork(obs_dim, act_dim).to(device)

    # -------- LOAD BC EXPERT --------
    print("Loading Behavioral Cloning expert weights...")
    q_net.load_state_dict(torch.load(BC_MODEL_PATH, map_location=device))
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPS_START
    global_step = 0

    print("Starting DQN + Imitation Learning training...")

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(MAX_STEPS):
            action = select_action(state, q_net, epsilon, act_dim)
            next_state, reward, done, truncated, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1

            # Train
            if len(replay_buffer) >= MIN_BUFFER_SIZE and global_step % TRAIN_FREQ == 0:
                loss = train_step(q_net, target_net, optimizer, replay_buffer)
                writer.add_scalar("train/loss", loss, global_step)

            # Update target network
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Logging (SB3-like)
        writer.add_scalar("rollout/ep_rew_mean", episode_reward, global_step)
        writer.add_scalar("rollout/ep_len_mean", episode_length, global_step)
        writer.add_scalar("train/epsilon", epsilon, global_step)

        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Length: {episode_length:4d} | "
            f"Epsilon: {epsilon:.3f}"
        )

        # Checkpoint
        if episode % 100 == 0:
            ckpt_path = os.path.join(MODEL_DIR, f"dqn_il_ep_{episode}.pth")
            torch.save(q_net.state_dict(), ckpt_path)

    # Save final model
    torch.save(q_net.state_dict(), os.path.join(MODEL_DIR, "dqn_il_final.pth"))
    env.close()
    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()