import os
import pickle
import numpy as np
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from trex_env_imitation import DinoEnv

# ======================================================
# Configuration
# ======================================================
ALGORITHM = "Behavioral_Cloning"
models_dir = "BC_0"
log_dir = "BC_0_tensorboard"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

# ======================================================
# Load Data
# ======================================================
expert_data_path = "expert_demos.pkl"

if not os.path.exists(expert_data_path):
    print(f"Error: Not found {expert_data_path}.")
    exit()

with open(expert_data_path, "rb") as f:
    trajectories = pickle.load(f)

obs_list = []
action_list = []

for episode in trajectories:
    for obs, action in episode:
        obs_list.append(obs)
        action_list.append(action)

obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
action_tensor = torch.tensor(action_list, dtype=torch.long)

dataset = TensorDataset(obs_tensor, action_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"{len(obs_list)} observations loaded.")

# ======================================================
# Model Definition
# ======================================================
class PolicyNetwork(nn.Module):
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

# Create env to get dimensions and for evaluation
env = DinoEnv(render_mode=None) # No render during training evaluation to be faster
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ======================================================
# Evaluation Function (Generates the graphs)
# ======================================================
def evaluate_policy(env, policy, n_eval_episodes=3):
    policy.eval()
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    policy.train() # Switch back to training mode
    return np.mean(episode_rewards), np.mean(episode_lengths)

# ======================================================
# Training Loop
# ======================================================
epochs = 1000
total_steps = 0
EVAL_FREQ = 20 # Evaluate every 20 epochs

print(f"Starting BC training for {epochs} epochs...")

for epoch in range(epochs):
    epoch_loss = 0
    for batch_obs, batch_action in dataloader:
        # Optimization
        logits = policy(batch_obs)
        loss = criterion(logits, batch_action)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_steps += 1
        epoch_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), total_steps)

    avg_loss = epoch_loss / len(dataloader)
    
    # --- EVALUATION STEP (This creates the graphs) ---
    if (epoch + 1) % EVAL_FREQ == 0:
        mean_reward, mean_length = evaluate_policy(env, policy)
        
        # Log exact same names as Stable Baselines 3
        writer.add_scalar("rollout/ep_rew_mean", mean_reward, total_steps)
        writer.add_scalar("rollout/ep_len_mean", mean_length, total_steps)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Reward={mean_reward:.1f} | Length={mean_length:.1f}")

    # Checkpointing
    if (epoch + 1) % 100 == 0:
        save_path = os.path.join(models_dir, f"bc_model_epoch_{epoch+1}.pth")
        torch.save(policy.state_dict(), save_path)

# Save Final
final_path = os.path.join(models_dir, "bc_final.pth")
torch.save(policy.state_dict(), final_path)
writer.close()
env.close()
print("Done!")