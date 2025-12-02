import pickle
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

from trex_env_imitation import DinoEnv

# --- Load demonstrations ---
expert_data_path = "expert_demos.pkl"
with open(expert_data_path, "rb") as f:
    trajectories = pickle.load(f)

# --- Prepare dataset ---
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

# --- Define the policy ---
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

env = DinoEnv()
policy = PolicyNetwork(input_dim=env.observation_space.shape[0],
                       output_dim=env.action_space.n)

optimizer = optim.Adam(policy.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --- Training ---
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for batch_obs, batch_action in dataloader:
        logits = policy(batch_obs)
        loss = criterion(logits, batch_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# --- Save model ---
torch.save(policy.state_dict(), "bc_dino_policy.pth")
print("Policy trained and saved as bc_dino_policy.pth")