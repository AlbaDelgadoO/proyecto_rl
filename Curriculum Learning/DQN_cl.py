import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from trex_env_cl import DinoEnv, CurriculumWrapper

# ======================================================
# Crear entorno con CurriculumWrapper y Monitor
# ======================================================
def make_env():
    env = DinoEnv()
    env = CurriculumWrapper(env)
    env = Monitor(env)
    return env

env = make_env()

# ======================================================
# Configuración
# ======================================================
n_steps = 300_000
checkpoint_dir = "./DQN_curriculum/"
tensorboard_log = "./DQN_curriculum_tensorboard/"
model_path = os.path.join(checkpoint_dir, "DQN_curriculum_model.zip")

os.makedirs(checkpoint_dir, exist_ok=True)

# ======================================================
# Crear o cargar modelo DQN
# ======================================================
if os.path.exists(model_path):
    print("Cargando modelo existente...")
    model = DQN.load(model_path, env=env, tensorboard_log=tensorboard_log)
else:
    print("Creando modelo nuevo...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log=tensorboard_log
    )

# ======================================================
# Callback para checkpoints cada 20k pasos
# ======================================================
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="DQN_curriculum"
)

# ======================================================
# Entrenamiento
# ======================================================
model.learn(total_timesteps=n_steps, callback=checkpoint_callback, progress_bar=True)

# Guardar modelo final
model.save(model_path)

env.close()