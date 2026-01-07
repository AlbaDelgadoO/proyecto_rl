from trex_env import DinoEnv


env = DinoEnv()


obs, info = env.reset()
done = False


while not done:
    action = env.action_space.sample()  
    obs, reward, done, truncated, info = env.step(action)
    env.render()


env.close()

