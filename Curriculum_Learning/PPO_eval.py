import pygame
from stable_baselines3 import PPO
import numpy as np

from trex_env_cl2 import DinoEnv

# ----------------------------------------
# Load trained model
# ----------------------------------------
model_path = "PPO_1/phase3_final.zip"
model = PPO.load(model_path)

# ----------------------------------------
# Create environment for rendering
# ----------------------------------------
env = DinoEnv(curriculum_phase=3)  

n_episodes = 5  

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        env.render()

    print(f"Episode {ep+1}: Total Reward = {total_reward}")

env.close()