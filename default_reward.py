import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_callback import CustomCallback

# Instantiate the custom environment
env = gym.make("InvertedPendulum-v5")

# Initialize PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1
)

# Initialize the callback
callback = CustomCallback()

# Train the model
model.learn(total_timesteps=100000, callback=callback)

# Plot rewards
plt.subplot(1, 2, 1)
plt.plot(callback.rewards)
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards')

# Plot total loss
plt.subplot(1, 2, 2)
plt.plot(callback.losses)
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Training Loss')

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.show()