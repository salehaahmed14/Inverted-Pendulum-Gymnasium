import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

# Define the custom callback
class CustomCallback(BaseCallback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.rewards = []
        self.losses = []  # Store total loss
        self.episode_reward = 0
        self.step_count = 0
        self.n_steps = 2048 

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


# Define the learning rates to test
learning_rates = [1e-5, 1e-4, 1e-3]

# Store results for plotting
all_rewards = {lr: [] for lr in learning_rates}
all_losses = {lr: [] for lr in learning_rates}

# Train and collect data for each learning rate
for lr in learning_rates:
    # Create the environment
    env = gym.make("InvertedPendulum-v5")
    
    # Initialize PPO model with varying learning rate
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,  # Varying the learning rate
        verbose=1,
        tensorboard_log="./ppo_inverted_pendulum_tensorboard/"
    )

    # Initialize the callback
    callback = CustomCallback()

    # Train the model
    model.learn(total_timesteps=100000, callback=callback)

    # Store results for plotting
    all_rewards[lr] = callback.rewards
    all_losses[lr] = callback.losses

    # Save the trained model for each learning rate
    model.save(f"ppo_inverted_pendulum_lr_{lr}")

# Plotting
plt.figure(figsize=(15, 15))

# Plot rewards
plt.subplot(2, 2, 1)
for lr in learning_rates:
    plt.plot(all_rewards[lr], label=f"Learning Rate {lr}")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Training Rewards for Different Learning Rates')
plt.legend()

# Plot total loss
plt.subplot(2, 2, 2)
for lr in learning_rates:
    plt.plot(all_losses[lr], label=f"Learning Rate {lr}")
plt.xlabel('Episodes')
plt.ylabel('Total Loss')
plt.title('Total Loss for Different Learning Rates')
plt.legend()

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top margin (lower value moves the plots down)

plt.show()
