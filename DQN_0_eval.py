import pygame
from stable_baselines3 import DQN
import numpy as np
import os
import sys

# Agregar el directorio de Curriculum_Learning al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Curriculum_Learning'))

# ----------------------------------------
# Cargar modelo entrenado
# ----------------------------------------
# Classic
# from Classic.trex_env import DinoEnv
# model_path = os.path.join("Classic", "DQN_fixed_model")
# model = DQN.load(model_path)

# Curriculum Learning
from trex_env_cl import DinoEnv

# Elige qué modelo cargar:
# model_path = os.path.join("Curriculum_Learning", "DQN_curriculum", "phase1_final")
# model_path = os.path.join("Curriculum_Learning", "DQN_curriculum", "phase2_final")
model_path = os.path.join("Curriculum_Learning", "DQN_curriculum", "phase3_final")  # Modelo final

model = DQN.load(model_path)

# ----------------------------------------
# Crear entorno para render
# ----------------------------------------
# Puedes elegir en qué fase evaluar (1, 2, o 3)
env = DinoEnv(curriculum_phase=3)  # Evaluar en la fase más difícil

n_episodes = 5  # Número de episodios de evaluación

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done:
        # Obtener acción del modelo
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Renderizar la ventana del juego
        env.render()

    print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Steps = {steps}")

env.close()