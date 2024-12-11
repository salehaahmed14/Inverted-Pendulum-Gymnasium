import gymnasium as gym
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

class RewardLoggingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        # Access reward directly from locals (more efficient)
        reward = self.locals.get('rewards')
        if reward is not None:
            self.rewards.append(reward)
        return True
    
# Create the environment
env = gym.make('InvertedPendulum-v5', render_mode="human")
env = Monitor(env)

# Create the DDPG model
model = DDPG('MlpPolicy', env, verbose=1)

# Create the callback
callback = RewardLoggingCallback()

# Train the model
model.learn(total_timesteps=1000, callback=callback)

# Render the environment
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.render()
env.close()

# Plot the training curve
plt.plot(callback.rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Inverted Pendulum DDPG Training Curve')
plt.show()