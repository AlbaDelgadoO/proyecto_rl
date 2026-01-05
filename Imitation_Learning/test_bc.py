import torch
import torch.nn as nn
import time
import os
import numpy as np
from trex_env_imitation import DinoEnv

# --- Redefine the PolicyNetwork Class ---
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

def main():
    # Setup Environment
    env = DinoEnv(render_mode="human")

    # Initialize Policy
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy = PolicyNetwork(input_dim, output_dim)

    # Load the saved model
    model_path = os.path.join("BC_0", "bc_model_epoch_1000.pth")

    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model not found at {model_path}. Did you run train_bc.py?")
        return

    policy.eval()

    # --- Evaluation Loop for 5 Episodes ---
    n_episodes = 5
    scores = []

    print(f"Starting evaluation for {n_episodes} episodes...")

    try:
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                env.render()
                
                # Convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # Get action from model
                with torch.no_grad():
                    logits = policy(obs_tensor)
                    action = torch.argmax(logits, dim=1).item()
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                
                # Optional: Adjust speed
                # time.sleep(0.01) 

            # Episode finished
            score = env.points
            scores.append(score)
            print(f"Episode {i+1}: Score = {score}")
            time.sleep(0.5) # Pause briefly between episodes

        # --- Final Results ---
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print("-" * 30)
        print(f"Results over {n_episodes} episodes:")
        print(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}")
        print("-" * 30)

    except KeyboardInterrupt:
        print("\nExiting...")

    env.close()

if __name__ == "__main__":
    main()