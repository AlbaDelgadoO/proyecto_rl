import torch
from trex_env_imitation import DinoEnv, HumanPolicy, Dinosaur
from train_bc import PolicyNetwork

env = DinoEnv()
policy = PolicyNetwork(input_dim=env.observation_space.shape[0],
                       output_dim=env.action_space.n)
policy.load_state_dict(torch.load("bc_dino_policy.pth"))
policy.eval()

obs, _ = env.reset()
done = False

while not done:
    env.render()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = policy(obs_tensor)
        action = torch.argmax(logits, dim=1).item()
    obs, reward, done, truncated, info = env.step(action)

env.close()
