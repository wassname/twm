
import gym
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any, Dict
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.play_craftax import CraftaxRenderer
from craftax.craftax.renderer import render_craftax_pixels, render_craftax_text, inverse_render_craftax_symbolic
from craftax.craftax.constants import Action
from craftax.craftax.craftax_state import EnvState
from gymnasium.wrappers import FrameStack, TimeLimit
from gymnax.wrappers import GymnaxToGymWrapper
import torch
from jaxtyping import Float
from torch import Tensor


def from_jax(t):
    return torch.as_tensor(t.tolist())

class CraftaxCompatWrapper(gym.Wrapper):
    """Misc compat"""
    
    def step(self, *args, **kwargs):
        next_obs, reward, terminated, truncated, info =  self.env.step(*args, **kwargs)
        return from_jax(next_obs), from_jax(reward), from_jax(terminated), from_jax(truncated), info
    
    def reset(self, *args, **kwargs):
        # if 'seed' in kwargs:
        #     kwargs['rng'] = jax.random.PRNGKey(int(kwargs['seed']))
        #     del kwargs['seed']
        obs, state = self.env.reset(*args, **kwargs)
        return from_jax(obs), state
    
    def get_action_meanings(self) -> Dict[int, str]:
        return {i:s for s,i in Action.__members__.items()}


class CraftaxRenderWrapper(gym.Wrapper):
    """
    Wrap Gymax (jas gym) to Gym (original gym)
    The main difference is that Gymax needs a rng key for every step and reset
    """
    def __init__(self, env, render_method:Optional[str]=None) -> None:
        super().__init__(env)
        self.render_method = render_method
        if render_method == 'play':
            self.renderer = CraftaxRenderer(self.env, self.env_params, pixel_render_size=1)
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
    
    def render(self, mode='rgb_array'):
        o = self.env.render()
        if self.renderer is not None:
            return self.renderer.render(self.env_state)
        elif self.render_method == 'text':            
            return render_craftax_text(self.env_state)
        else:
            return render_craftax_pixels(self.env_state)
        return o
        
    def close(self):
        if self.renderer is not None:
            self.renderer.pygame.quit()
            self.renderer.close()
        

        
def create_craftax_env(game, noop_max=30, frame_skip=4, frame_stack=4, frame_size=84,
                     episodic_lives=True, grayscale=True, time_limit=27000, seed=42, eval=False, num_envs=1):
    """
    Craftax with
    - frame_stack 4?
    time_limit = 27000
    
    """
    game = "Craftax-Symbolic-v1"
    # see https://github.dev/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py
    # env = make_craftax_env_from_name(game, auto_reset=eval)
    env = make_craftax_env_from_name(game, auto_reset=True)
    env = GymnaxToGymWrapper(env, env.default_params, seed=seed) 
    # env = LogWrapper(env)
    # FIXME: sort this out
    # if not eval:
    #     env = OptimisticResetVecEnvWrapper(
    #         env,
    #         num_envs=num_envs,
    #         reset_ratio=min(not eval, True),
    #     )
    # else:
    #     env = BatchEnvWrapper(env, num_envs=num_envs)
    env = CraftaxRenderWrapper(env, render_method=None)
    env = CraftaxCompatWrapper(env)
    env = gym.wrappers.FrameStack(env, frame_stack)
    
    # env = make_craftax_env_from_name(game, auto_reset=True)
    # env = Gymax2GymWrapper(env, render_method=None)
    
    env.unwrapped.spec = gym.spec(game) # required for AtariPreprocessing
    env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
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
        

def craftax_symobs_to_img(obs: Float[Tensor, 'batch ... 8268'], real_env_state: EnvState) -> Float[Tensor, 'batch ... h w c']:
    """convert symbolic obs to image"""
    assert obs.shape[-1]==8268
    obs2 = obs.reshape(-1, 8268).cpu().numpy()
    env_state = [inverse_render_craftax_symbolic(oo, env_state=real_env_state) for oo in obs2]
    im = [render_craftax_pixels(oo, block_pixel_size=10).astype(np.uint8) for oo in env_state]
    im =  torch.from_numpy(np.array(im))

    # return to original shape
    im = im.reshape(*obs.shape[:-1], *im.shape[-3:])
    return im


def create_vector_env(num_envs, env):
    # TODO: wait isn't this meant to be used before gymax2gymnasium?
    return BatchEnvWrapper(env, num_envs=num_envs)
