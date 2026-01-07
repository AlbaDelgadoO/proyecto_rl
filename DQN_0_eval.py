import pygame
from stable_baselines3 import DQN
import numpy as np
import os
import sys

# Agregate the directory of Curriculum_Learning to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Curriculum_Learning'))

# ----------------------------------------
# Load trained model
# ----------------------------------------
# Classic
# from Classic.trex_env import DinoEnv
# model_path = os.path.join("Classic", "DQN_fixed_model")
# model = DQN.load(model_path)

# Curriculum Learning
from trex_env_cl import DinoEnv

# Choose which model to load:
# model_path = os.path.join("Curriculum_Learning", "DQN_curriculum", "phase1_final")
# model_path = os.path.join("Curriculum_Learning", "DQN_curriculum", "phase2_final")
# model_path = os.path.join("Curriculum_Learning", "DQN_curriculum", "phase3_final")  # Final model
model_path = os.path.join("Curriculum_Learning", "DQN_0_curriculum", "phase3_final")  # Final model
model = DQN.load(model_path)

# ----------------------------------------
# Create environment for rendering
# ----------------------------------------
# You can choose which phase to evaluate (1, 2, or 3)
env = DinoEnv(curriculum_phase=3)  # Evaluate in the hardest phase

n_episodes = 5  # Number of evaluation episodes

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done:
        # Get action from the model
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Render the game window
        env.render()

    print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Steps = {steps}")

env.close()