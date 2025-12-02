import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from trex_env import DinoEnv

# ======================================================
# Crear entorno con Monitor
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
# Crear modelo DQN compatible con tu versión de SB3
# ======================================================
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,

    learning_rate=5e-5,
    buffer_size=150_000,
    learning_starts=5_000,
    batch_size=128,

    gamma=0.99,
    tau=1.0,
    train_freq=1,
    target_update_interval=1_000,

    exploration_fraction=0.15,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,

    tensorboard_log=tensorboard_log,

    # Aquí NO va "dueling=True". Para tu versión NO existe ese parámetro.
    # Para dueling, tu versión de SB3 lo incluye automáticamente
    # dentro de la arquitectura por defecto.
    policy_kwargs=dict(
        net_arch=[512, 512, 256]
    )
)

# ======================================================
# Callback para checkpoints
# ======================================================
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="DQN_0"
)

# ======================================================
# Entrenamiento
# ======================================================
model.learn(
    total_timesteps=n_steps,
    callback=checkpoint_callback,
    progress_bar=True
)

model.save("DQN_fixed_model")

env.close()