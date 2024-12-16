from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

#define clip_range values
clip_range_values = [0.1, 0.2, 0.3]

#result variables for plotting
all_rewards = {clip: [] for clip in clip_range_values}
all_losses = {clip: [] for clip in clip_range_values}

# train and collect data for each clip_range
for clip_range in clip_range_values:
    # instantiate the custom environment
    env = CustomInvertedPendulum()

    #initialize PPO model with default parameters and varying clip_range
    model = PPO(
        policy="MlpPolicy",
        env=env,
        clip_range=clip_range,
        verbose=1
    )

    # initialize the callback
    callback = CustomCallback()

    # train the model
    model.learn(total_timesteps=100000, callback=callback)

    #store results for plotting
    all_rewards[clip_range] = callback.rewards
    all_losses[clip_range] = callback.losses

# plot episodic rewards
plt.subplot(1, 2, 1)
for clip_range in clip_range_values:
    plt.plot(all_rewards[clip_range], label=f"Clip Range {clip_range}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different Clip Range Values')
plt.legend()

#plot total loss
plt.subplot(1, 2, 2)
for clip_range in clip_range_values:
    plt.plot(all_losses[clip_range], label=f"Clip Range {clip_range}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different Clip Range Values')
plt.legend()

plt.tight_layout()
plt.show()