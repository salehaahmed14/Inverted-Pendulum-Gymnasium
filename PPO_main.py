import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# Optimal hyperparameters
gamma = 0.99
learning_rate = 0.001
n_steps = 2048
clip_range = 0.3
    
# Instantiate the custom environment
env = CustomInvertedPendulum()

# Initialize PPO model with optimal parameters
model = PPO(
    policy="MlpPolicy",
    env=env,
    gamma=gamma,
    n_steps=n_steps,
    learning_rate=learning_rate,
    clip_range=clip_range,
    verbose=1
)

# Initialize the callback
callback = CustomCallback(n_steps)

# Train the model
model.learn(total_timesteps=100000, callback=callback)

# save the model
model.save("ppo_inverted_pendulum")

# Plot rewards
plt.subplot(1, 2, 1)
plt.plot(callback.rewards, label="Episodic Rewards")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards with Optimal Hyperparameters')
plt.legend()

# Plot total loss
plt.subplot(1, 2, 2)
plt.plot(callback.losses, label="Total Loss Per Update")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Training Loss with Optimal Hyperparameters')
plt.legend()

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.show()