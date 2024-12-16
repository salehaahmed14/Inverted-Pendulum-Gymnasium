import gymnasium as gym

#custom Gym wrapper for the InvertedPendulum
class CustomInvertedPendulum(gym.Wrapper):
    def __init__(self, render_mode=None):
        super(CustomInvertedPendulum, self).__init__(gym.make("InvertedPendulum-v5", render_mode=render_mode))

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        cart_position = obs[0]  #cart position
        theta = obs[1]  #pendulum angle
        angular_velocity = obs[3]  #angular velocity

        # thresholds
        max_theta = 0.2
        max_position = 2
        max_angular_velocity = 5

        # reward components: 0 (worst) to 1 (best)
        angle_reward = max(0, 1 - abs(theta) / max_theta)
        position_reward = max(0, 1 - abs(cart_position) / max_position)
        velocity_reward = max(0, 1 - abs(angular_velocity) / max_angular_velocity)

        #combine all components with weights
        shaped_reward = (
            0.6 * angle_reward +       # 60% weight to angle control
            0.3 * position_reward +    # 30% weight to cart position
            0.1 * velocity_reward      # 10% weight to smooth angular control
        )

        #normalize reward in the range [0, 1]
        reward = max(0, min(1, shaped_reward))

        return obs, reward, done, truncated, info