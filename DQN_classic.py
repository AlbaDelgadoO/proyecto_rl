import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from trex_env import DinoEnv

# ======================================================
# Crear entorno con Monitor para TensorBoard
# ======================================================
def make_env():
    env = DinoEnv()      
    env = Monitor(env)   
    return env

# ======================================================
# Configuración
# ======================================================
n_steps = 300_000
checkpoint_dir = "./DQN_0/"
tensorboard_log = "./DQN_0_tensorboard/"

os.makedirs(checkpoint_dir, exist_ok=True)

env = make_env()

# ======================================================
# Crear modelo DQN
# ======================================================
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=5_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    exploration_initial_eps=1.0,
    tensorboard_log=tensorboard_log
)

# ======================================================
# Callback para checkpoints cada 20k pasos
# ======================================================
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="DQN_0"
)

# ======================================================
# Entrenamiento
# ======================================================
model.learn(total_timesteps=n_steps, callback=checkpoint_callback, progress_bar=True)

model.save("DQN_0_model")

env.close()