import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

class CustomCallback(BaseCallback):
    def __init__(self, n_steps):  # Pass n_steps as an argument
        super(CustomCallback, self).__init__()
        self.rewards = []
        self.losses = []  # Store total loss
        self.episode_reward = 0
        self.step_count = 0
        self.n_steps = n_steps  # Use the passed n_steps value

    def _on_step(self) -> bool:
        reward = self.locals.get('rewards')  # Get the reward at the current step
        done = self.locals.get('dones')  # Get the done status (whether the episode ended)
        
        if reward is not None:
            self.episode_reward += reward
        
        self.step_count += 1
        
        # Check if we should log the loss after n_steps steps
        if self.step_count % self.n_steps == 0:
            # Log the total loss (sum of all individual losses)
            total_loss = self.model.logger.name_to_value.get('train/loss', None)
            
            if total_loss is not None:
                self.losses.append(total_loss)

        if done:  # When an episode is done, store the reward and reset
            self.rewards.append(self.episode_reward)
            self.episode_reward = 0  # Reset episode reward for the next episode

        return True

# Define n_steps values for search
n_steps_values = [64, 128, 256, 512]

# Store results for plotting
all_rewards = {n_steps: [] for n_steps in n_steps_values}
all_losses = {n_steps: [] for n_steps in n_steps_values}

# Train and collect data for each n_steps
for n_steps in n_steps_values:
    # Create the environment
    env = gym.make("InvertedPendulum-v5")
    
    # Initialize PPO model with default parameters and varying n_steps
    model = PPO(
        policy="MlpPolicy",
        env=env,
        gamma=0.99,  # default value
        learning_rate=0.0003,  # default value
        n_steps=n_steps,  # Set the current value of n_steps
        batch_size=n_steps,  # Set BATCH_SIZE = n_steps
        verbose=1,
        tensorboard_log="./ppo_inverted_pendulum_tensorboard/"
    )

    # Initialize the callback with the current n_steps value
    callback = CustomCallback(n_steps=n_steps)

    # Train the model
    model.learn(total_timesteps=100000, callback=callback)

    # Store results for plotting
    all_rewards[n_steps] = callback.rewards
    all_losses[n_steps] = callback.losses

    # Save the trained model for each n_steps
    model.save(f"ppo_inverted_pendulum_n_steps_{n_steps}")

# Plotting
plt.figure(figsize=(15, 15))

# Plot rewards
plt.subplot(2, 2, 1)
for n_steps in n_steps_values:
    plt.plot(all_rewards[n_steps], label=f"n_steps {n_steps}")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Training Rewards for Different n_steps Values')
plt.legend()

# Plot total loss
plt.subplot(2, 2, 2)
for n_steps in n_steps_values:
    plt.plot(all_losses[n_steps], label=f"n_steps {n_steps}")
plt.xlabel('Episodes')
plt.ylabel('Total Loss')
plt.title('Total Loss for Different n_steps Values')
plt.legend()

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top margin (lower value moves the plots down)

plt.show()
