import gymnasium as gym

# Custom Gym wrapper for the InvertedPendulum
class CustomInvertedPendulum(gym.Wrapper):
    def __init__(self, env):
        super(CustomInvertedPendulum, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        cart_position = obs[0]  # Cart position
        theta = obs[1]  # Pendulum angle
        theta_dot = obs[3]  # Angular velocity

        # Reward shaping
        max_theta = 0.2  # Threshold for the pendulum angle
        max_position = 2  # Threshold for cart position

        # Normalize angle reward: 0 (worst) to 1 (best)
        angle_reward = max(0, 1 - abs(theta) / max_theta)

        # Normalize position reward: 0 (worst) to 1 (best)
        position_reward = max(0, 1 - abs(cart_position) / max_position)

        # Normalize velocity penalty: Penalize large angular velocities
        max_theta_dot = 5  # Maximum expected angular velocity (adjust if needed)
        velocity_penalty = max(0, 1 - abs(theta_dot) / max_theta_dot)

        # Combine all components with weights
        shaped_reward = (
            0.6 * angle_reward +       # 60% weight to angle control
            0.3 * position_reward +    # 30% weight to cart position
            0.1 * velocity_penalty     # 10% weight to smooth angular control
        )

        # Ensure reward is in the range [0, 1]
        reward = max(0, min(1, shaped_reward))
        return obs, reward, done, truncated, info