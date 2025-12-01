import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

from trex_env_cl import DinoEnv

# ======================================================
# Entorno wrapper para curriculum learning
# ======================================================
class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CurriculumWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Cambiar dificultad según puntos
        points = getattr(self.env, "points", 0)
        if points < 500:
            self.env.difficulty = 1
        elif points < 1500:
            self.env.difficulty = 2
        else:
            self.env.difficulty = 3

        return obs, reward, done, truncated, info

# ======================================================
# Crear entorno con Monitor para TensorBoard
# ======================================================
def make_env():
    env = DinoEnv()
    env = CurriculumWrapper(env)
    env = Monitor(env)
    return env

# ======================================================
# Configuración
# ======================================================
n_steps = 300_000
checkpoint_dir = "./PPO_curriculum/"
tensorboard_log = "./PPO_curriculum_tensorboard/"
model_path = os.path.join(checkpoint_dir, "PPO_curriculum_model.zip")

os.makedirs(checkpoint_dir, exist_ok=True)

env = make_env()

# ======================================================
# Crear o cargar modelo PPO
# ======================================================
if os.path.exists(model_path):
    print("Cargando modelo existente...")
    model = PPO.load(model_path, env=env, tensorboard_log=tensorboard_log)
else:
    print("Creando modelo nuevo...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=tensorboard_log
    )

# ======================================================
# Callback para checkpoints cada 20k pasos
# ======================================================
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="PPO_curriculum"
)

# ======================================================
# Entrenamiento
# ======================================================
model.learn(total_timesteps=n_steps, callback=checkpoint_callback, progress_bar=True)

# Guardar modelo final
model.save(model_path)

env.close()
