import torch
import torch.nn as nn
import time
import os
import numpy as np

from trex_env import DinoEnv  


# ======================================================
# Q-Network (SAME architecture as BC and DQN_IL)
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


def main():
    # --------------------------------------------------
    # Environment
    # --------------------------------------------------
    env = DinoEnv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --------------------------------------------------
    # Load trained DQN + IL model
    # --------------------------------------------------
    model_path = os.path.join("DQN_IL_models", "dqn_il_final.pth")

    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        return

    q_net = QNetwork(obs_dim, act_dim)
    q_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    q_net.eval()

    print(f"Loaded model from {model_path}")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    n_episodes = 5
    scores = []

    print(f"Evaluating DQN + IL for {n_episodes} episodes...")

    try:
        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            done = False

            while not done:
                env.render()

                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = q_net(obs_t)
                    action = torch.argmax(q_values, dim=1).item()

                obs, reward, done, truncated, info = env.step(action)

            score = env.points
            scores.append(score)
            print(f"Episode {ep}: Score = {score}")
            time.sleep(0.5)

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        print("-" * 30)
        print(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}")
        print("-" * 30)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    env.close()


if __name__ == "__main__":
    main()