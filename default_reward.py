import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class CustomCallback(BaseCallback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.rewards = []
        self.losses = []
        self.episode_reward = 0
        self.step_count = 0

    def _on_step(self) -> bool:
        reward = self.locals.get('rewards')  # Get the reward at the current step
        done = self.locals.get('dones')  # Get the done status (whether the episode ended)
        
        if reward is not None:
            self.episode_reward += reward
        
        self.step_count += 1
        
        # Check if we should log the loss after n_steps steps
        if self.step_count % 2048 == 0:
            # Log the total loss (sum of all individual losses)
            total_loss = self.model.logger.name_to_value['train/loss']
            
            if total_loss is not None:
                self.losses.append(total_loss)

        if done:  # When an episode is done, store the reward and reset
            self.rewards.append(self.episode_reward)
            self.episode_reward = 0  # Reset episode reward for the next episode

        return True

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