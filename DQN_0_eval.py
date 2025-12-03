import pygame
from stable_baselines3 import DQN
import numpy as np
import os

from Classic.trex_env import DinoEnv

# ----------------------------------------
# Cargar modelo entrenado
# ----------------------------------------
# Ruta al modelo dentro de Classic
model_path = os.path.join("Classic", "DQN_fixed_model")
model = DQN.load(model_path)


# ----------------------------------------
# Crear entorno para render
# ----------------------------------------
env = DinoEnv()  # Aquí necesitamos que render() funcione con pygame

n_episodes = 5  # Número de episodios de evaluación

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        # Obtener acción del modelo
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Renderizar la ventana del juego
        env.render()

    print(f"Episode {ep+1}: Total Reward = {total_reward}")

env.close()