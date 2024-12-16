from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from custom_env import CustomInvertedPendulum

env = Monitor(CustomInvertedPendulum(render_mode='human'))

#load the pre-trained model
model = PPO.load("ppo_inverted_pendulum", env=env)

#evaluate the trained policy
mean_reward, std_reward = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=10, 
    deterministic=True, 
    render=True
)

print(f"Mean reward over 10 episodes: {mean_reward:.2f}")
print(f"Standard deviation of rewards: {std_reward:.2f}")
