import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

# Custom callback to log rewards and losses
class CustomCallback(BaseCallback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def _on_step(self) -> bool:
        # Log rewards and losses only at the end of an episode
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                # Log episode reward
                self.rewards.append(info["episode"]["r"])

                # Log losses at the end of the episode
                policy_loss = float(self.model.logger.name_to_value.get("train/policy_loss", 0))
                value_loss = float(self.model.logger.name_to_value.get("train/value_loss", 0))
                entropy_loss = float(self.model.logger.name_to_value.get("train/entropy_loss", 0))
                self.policy_losses.append(policy_loss)
                self.value_losses.append(value_loss)
                self.entropy_losses.append(entropy_loss)

        return True

# Create the environment
env = gym.make("InvertedPendulum-v5")

# Set the A2C parameters
gamma = 0.989935318520895
learning_rate = 0.009037472400690326
n_steps = 512  # A2C typically uses smaller n_steps

# Initialize the A2C model
model = A2C(
    policy="MlpPolicy",
    env=env,
    gamma=gamma,
    learning_rate=learning_rate,
    n_steps=n_steps,
    verbose=1,
    tensorboard_log="./a2c_inverted_pendulum_tensorboard/"
)

# Initialize the callback
callback = CustomCallback()

# Train the model
model.learn(total_timesteps=70000, callback=callback)

# Save the trained model
model.save("a2c_inverted_pendulum")

# Plotting
plt.figure(figsize=(15, 10))

# Plot rewards
plt.subplot(2, 2, 1)
plt.plot(callback.rewards, label="Rewards", color="blue")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Training Rewards')

# Plot policy loss
plt.subplot(2, 2, 2)
plt.plot(callback.policy_losses, label="Policy Loss", color="orange")
plt.xlabel('Episodes')
plt.ylabel('Policy Loss')
plt.title('Policy Gradient Loss')

# Plot value loss
plt.subplot(2, 2, 3)
plt.plot(callback.value_losses, label="Value Loss", color="green")
plt.xlabel('Episodes')
plt.ylabel('Value Loss')
plt.title('Value Function Loss')

# Plot entropy loss
plt.subplot(2, 2, 4)
plt.plot(callback.entropy_losses, label="Entropy Loss", color="red")
plt.xlabel('Episodes')
plt.ylabel('Entropy Loss')
plt.title('Entropy Loss')

plt.tight_layout()
plt.show()
