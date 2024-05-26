import gymnasium


class NoAutoReset(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.final_observation = None
        self.final_info = None

    def reset(self, seed=None, options=None):
        if self.final_observation is None or (
            options is not None and options.get("force", False)
        ):
            return self.env.reset(seed=seed, options=options)
        return self.final_observation, self.final_info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.final_observation = obs
            self.final_info = info
        return obs, reward, terminated, truncated, info
