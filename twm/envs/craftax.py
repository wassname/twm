# import gym
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any, Dict
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.play_craftax import CraftaxRenderer
from craftax.craftax.renderer import (
    render_craftax_pixels,
    render_craftax_text,
    inverse_render_craftax_symbolic,
)
from craftax.craftax.constants import Action
from craftax.craftax.craftax_state import EnvState
import gymnasium
from gymnasium.wrappers.jax_to_numpy import numpy_to_jax, jax_to_numpy
from gymnasium.wrappers.jax_to_torch import jax_to_torch, torch_to_jax
from gymnasium.wrappers.numpy_to_torch import numpy_to_torch, torch_to_numpy
from gymnasium.wrappers import FrameStackObservation, TimeLimit, JaxToTorch
import gymnasium.spaces as gym_spaces
from twm.envs.gymnax2gymnasium import GymnaxToGymWrapper, GymnaxToVectorGymWrapper
import torch
from jaxtyping import Float, Int, Bool
from torch import Tensor
from twm.custom_types import Obs, TrcBool, TrcFloat, TrceInt

import gymnasium.spaces as gym_spaces
from gymnasium.wrappers import TransformObservation


def permute_env(env, prm=[1, 0, 2]):
    os = env.observation_space
    oshape = os.shape
    new_os = gym_spaces.Box(
        low=np.transpose(os.low, prm),
        high=np.transpose(os.high, prm),
        shape=[oshape[i] for i in prm],
        dtype=os.dtype,
    )
    env = TransformObservation(env, lambda x: jnp.transpose(x, prm), obs_space=new_os)
    return env

def to_torch(v) -> torch.Tensor:
    if isinstance(v, jnp.ndarray):
        if v.dtype=='bool':
            # bool doesn't convert using the jax_to_torch dlpack
            # return torch.from_numpy(v._npy_value.copy())
            return torch.as_tensor(v.tolist())
        return jax_to_torch(v)
    if isinstance(v, np.ndarray):
        return numpy_to_torch(v)
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.as_tensor(v)



class CraftaxCompatWrapper(gymnasium.core.Wrapper):
    """
    Misc compat
    - from jax
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self._env = env.unwrapped._env

    def step(
        self, action: int
    ) -> Tuple[Float[Tensor, "frames odim"], float, bool, bool, Dict]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            numpy_to_torch(next_obs).to(torch.float16), # in symbolic only lighting needs values other than 0 and 1
            to_torch(reward),
            to_torch(terminated),
            to_torch(truncated),
            info,
        )

    def reset(self, *args, **kwargs):
        obs, state = self.env.reset(*args, **kwargs)
        return numpy_to_torch(obs), state

    def get_action_meanings(self) -> Dict[int, str]:
        return {i.value: s for s, i in Action.__members__.items()}

    @property
    def env_state(self):
        return self.env.unwrapped.env_state


class CraftaxRenderWrapper(gymnasium.core.Wrapper):
    """
    Wrap Gymax (jas gym) to Gym (original gym)
    The main difference is that Gymax needs a rng key for every step and reset
    """

    def __init__(self, env, render_method: Optional[str] = None) -> None:
        super().__init__(env)
        self.render_method = render_method
        if render_method == "play":
            self.renderer = CraftaxRenderer(
                self.env, self.env_params, pixel_render_size=1
            )
        self.renderer = None

    def step(self, *args, **kwargs):
        o = self.env.step(*args, **kwargs)
        if self.renderer is not None:
            self.renderer.update()
        return o

    def reset(self, *args, **kwargs):
        o = self.env.reset(*args, **kwargs)
        if self.renderer is not None:
            self.renderer.update()
        return o

    def render(self, mode="rgb_array"):
        o = self.env.render()
        if self.renderer is not None:
            return self.renderer.render(self.env_state)
        elif self.render_method == "text":
            return render_craftax_text(self.env_state)
        else:
            return render_craftax_pixels(self.env_state)
        return o

    def close(self):
        if self.renderer is not None:
            self.renderer.pygame.quit()
            self.renderer.close()


def create_craftax_env(
    game, frame_stack=4, time_limit=27000, seed=42, eval=False, num_envs=1
):
    """
    Craftax with
    - frame_stack 4?
    time_limit = 27000

    """
    game = "Craftax-Symbolic-v1"
    # see https://github.dev/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py
    env = make_craftax_env_from_name(game, auto_reset=not eval)
    if num_envs > 1:
        # FIXME: naive optimistic resets don't work well with multiple envs see OptimisticResetVecEnvWrapper
        env = GymnaxToVectorGymWrapper(env, seed=seed, num_envs=num_envs)
        raise NotImplementedError("Only num_envs > 1 supported FIXME")
    else:
        env = GymnaxToGymWrapper(env, env.default_params, seed=seed)
    # env = LogWrapper(env)

    # We have to vectorise using jax earlier as there is not framestack wrapepr avaiable for jax
    # but then the framestack dim is before the env dim [framestack, batch, obs_dim] so lets swap those
    env = FrameStackObservation(env, frame_stack)
    if num_envs > 1:
        env = permute_env(env, [1, 0, 2])

    # env.unwrapped.spec = gym.spec(game) # required for AtariPreprocessing
    if not eval:
        env = TimeLimit(env, max_episode_steps=time_limit)

    env = CraftaxRenderWrapper(env, render_method=None)
    env = CraftaxCompatWrapper(env)
    return env


"""
Below files from https://github.dev/MichaelTMatthews/Craftax_Baselines/wrappers.py
"""


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(rng, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


def craftax_symobs_to_img(
    obs: Float[Tensor, "batch ... 8268"], real_env_state: EnvState
) -> Float[Tensor, "batch ... h w c"]:
    """convert symbolic obs to image"""
    assert obs.shape[-1] == 8268
    obs2 = obs.reshape(-1, 8268).cpu().numpy()
    env_state = [
        inverse_render_craftax_symbolic(oo, env_state=real_env_state) for oo in obs2
    ]
    im = [
        render_craftax_pixels(oo, block_pixel_size=10).astype(np.uint8)
        for oo in env_state
    ]
    im = torch.from_numpy(np.array(im))

    # return to original shape
    im = im.reshape(*obs.shape[:-1], *im.shape[-3:])
    # [batch, h=130, w=110, c=3]
    return im


def create_vector_env(num_envs, env):
    # TODO: wait isn't this meant to be used before gymax2gymnasium?
    return BatchEnvWrapper(env, num_envs=num_envs)


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
