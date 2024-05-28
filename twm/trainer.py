import multiprocessing
import os
import time

import numpy as np
import torch
import torchvision
from PIL import ImageDraw
from loguru import logger
import wandb
from tqdm.auto import tqdm
from twm.agent import Agent, Dreamer
from twm.replay_buffer import ReplayBuffer
from twm import utils, metrics
from twm.envs.craftax import (
    create_craftax_env,
    craftax_symobs_to_img,
    create_vector_env,
    NoAutoReset,
)
from einops import rearrange, repeat, reduce

mininterval = 30
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class Trainer:
    def __init__(self, config):
        if config["buffer_prefill"] <= 0:
            raise ValueError()
        if config["pretrain_obs_p"] + config["pretrain_dyn_p"] > 1:
            raise ValueError()

        self.config = config
        self.env = self._create_env_from_config(config)
        self.replay_buffer = ReplayBuffer(config, self.env)

        num_actions = self.env.action_space.n
        self.action_meanings = self.env.get_action_meanings()
        self.agent = Agent(config, num_actions).to(config["model_device"])

        # metrics that won't be summarized, the last value will be used instead
        except_keys = ["buffer/size", "buffer/total_reward", "buffer/num_episodes"]
        self.summarizer = metrics.MetricsSummarizer(except_keys=except_keys)
        self.last_eval = 0
        self.total_eval_time = 0

    def print_stats(self):
        count_params = lambda module: sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        agent = self.agent
        wm = agent.wm
        ac = agent.ac
        print("# Parameters")
        print("Observation model:", count_params(wm.obs_model))
        print("Dynamics model:", count_params(wm.dyn_model))
        print("Actor:", count_params(ac.actor_model))
        print("Critic:", count_params(ac.critic_model))
        print("World model:", count_params(wm))
        print("Actor-critic:", count_params(ac))
        print(
            "Observation encoder + actor:",
            count_params(wm.obs_model.encoder) + count_params(ac.actor_model),
        )
        print("Total:", count_params(agent))

    def close(self):
        self.env.close()

    @staticmethod
    def _create_env_from_config(config, eval=False):
        # env = create_atari_env(
        #     config['game'], noop_max, config['env_frame_skip'], config['env_frame_stack'],
        #     config['env_frame_size'], config['env_episodic_lives'], config['env_grayscale'], config['env_time_limit'])
        env = create_craftax_env(
            config["game"],
            frame_stack=config["env_frame_stack"],
            time_limit=config["env_time_limit"],
            eval=eval,
        )
        if eval:
            # FIXME: make it work for crafter
            env = NoAutoReset(env)  # must use options={'force': True} to really reset
        return env

    def _create_buffer_obs_policy(self):
        # actor-critic policy acting on buffer data at index
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        ac = agent.ac
        replay_buffer = self.replay_buffer
        dreamer = None

        @torch.no_grad()
        def policy(index):
            nonlocal dreamer
            if dreamer is None:
                prefix = config["wm_memory_length"] - 1
                start_o = replay_buffer.get_obs(
                    [[index]], device=device, prefix=prefix + 1
                )
                start_a = replay_buffer.get_actions(
                    [[index - 1]], device=device, prefix=prefix
                )
                start_r = replay_buffer.get_rewards(
                    [[index - 1]], device=device, prefix=prefix
                )
                start_terminated = replay_buffer.get_terminated(
                    [[index - 1]], device=device, prefix=prefix
                )
                start_truncated = replay_buffer.get_truncated(
                    [[index - 1]], device=device, prefix=prefix
                )

                dreamer = Dreamer(config, wm, mode="observe", ac=ac, store_data=False)
                dreamer.observe_reset(
                    start_o, start_a, start_r, start_terminated, start_truncated
                )
                a = dreamer.act()
            else:
                o = replay_buffer.get_obs([[index]], device=device)
                a = replay_buffer.get_actions([[index - 1]], device=device)
                r = replay_buffer.get_rewards([[index - 1]], device=device)
                terminated = replay_buffer.get_terminated([[index - 1]], device=device)
                truncated = replay_buffer.get_truncated([[index - 1]], device=device)
                dreamer.observe_step(a, o, r, terminated, truncated)
                a = dreamer.act()
            return a.squeeze().item()

        return policy

    def _create_start_z_sampler(self, temperature):
        obs_model = self.agent.wm.obs_model
        replay_buffer = self.replay_buffer

        @torch.no_grad()
        def sampler(n):
            device = next(obs_model.parameters()).device
            idx = utils.random_choice(
                replay_buffer.size, n, device=replay_buffer.device
            )
            o = replay_buffer.get_obs(idx, device=device).unsqueeze(1)
            z = obs_model.eval().encode_sample(o, temperature=temperature).squeeze(1)
            return z

        return sampler

    def run(self):
        config = self.config
        replay_buffer = self.replay_buffer

        log_every = 200
        self.last_eval = 0
        self.total_eval_time = 0

        logger.info("prefill the buffer with randomly collected data")
        random_policy = lambda index: replay_buffer.sample_random_action()
        for _ in tqdm(range(config["buffer_prefill"] - 1), mininterval=mininterval):
            replay_buffer.step(random_policy)
            metrics_d = {}
            metrics.update_metrics(metrics_d, replay_buffer.metrics(), prefix="buffer/")
            self.summarizer.append(metrics_d)
            if replay_buffer.size % log_every == 0:
                wandb.log(self.summarizer.summarize())

        # final prefill step
        replay_buffer.step(random_policy)
        metrics_d = {}
        metrics.update_metrics(metrics_d, replay_buffer.metrics(), prefix="buffer/")

        # pretrain on the prefilled data
        self._pretrain()

        eval_metrics = self._evaluate(is_final=False)
        metrics_d.update(eval_metrics)
        self.summarizer.append(metrics_d)
        wandb.log(self.summarizer.summarize())

        budget = config["budget"] - config["pretrain_budget"]
        budget_per_step = 0
        budget_per_step += (
            config["wm_train_steps"]
            * config["wm_batch_size"]
            * config["wm_sequence_length"]
        )
        budget_per_step += config["ac_batch_size"] * config["ac_horizon"]
        num_batches = budget / budget_per_step
        train_every = (replay_buffer.capacity - config["buffer_prefill"]) / num_batches

        # FIXME this just trains for buffer capacity not train budget?
        step_counter = 0
        logger.info(f"collect data in real environment train_every={train_every}, num_batches={num_batches} budget_per_step={budget_per_step}")
        with tqdm(
            total=replay_buffer.capacity - replay_buffer.size,
            unit="step",
            desc="train",
            mininterval=mininterval,
        ) as pbar:
            while replay_buffer.size < replay_buffer.capacity:
                collect_policy = self._create_buffer_obs_policy()
                should_log = False
                # logger.debug(f"step:0 collect {train_every} data")
                while (
                    step_counter <= train_every
                    and replay_buffer.size < replay_buffer.capacity
                ):
                    if replay_buffer.size - self.last_eval >= config["eval_every"]:
                        metrics_d = self._evaluate(is_final=False)
                        metrics.update_metrics(
                            metrics_d, replay_buffer.metrics(), prefix="buffer/"
                        )
                        self.summarizer.append(metrics_d)
                        wandb.log(self.summarizer.summarize())

                    replay_buffer.step(collect_policy)
                    step_counter += 1
                    pbar.update(1)

                    if replay_buffer.size % log_every == 0:
                        should_log = True
                # logger.debug(f"step:1")

                # train world model and actor-critic
                # logger.debug(f"train:0 train world model and actor-critic")
                metrics_hist = []
                while step_counter >= train_every:
                    step_counter -= train_every
                    metrics_d = self._train_step()
                    metrics_hist.append(metrics_d)
                # logger.debug(f"train:1")

                metrics_d = metrics.mean_metrics(metrics_hist)
                metrics.update_metrics(
                    metrics_d, replay_buffer.metrics(), prefix="buffer/"
                )
                # logger.debug(f"eval:0 evaluate")

                # evaluate
                if (
                    replay_buffer.size - self.last_eval >= config["eval_every"]
                    and replay_buffer.size < replay_buffer.capacity
                ):
                    eval_metrics = self._evaluate(is_final=False)
                    metrics_d.update(eval_metrics)
                    should_log = True
                # logger.debug(f"eval:1")

                self.summarizer.append(metrics_d)
                if should_log:
                    s = self.summarizer.summarize()
                    wandb.log(s)
                    logger.debug(s)
                

        logger.info("final evaluation")
        metrics_d = self._evaluate(is_final=True)
        metrics.update_metrics(metrics_d, replay_buffer.metrics(), prefix="buffer/")
        self.summarizer.append(metrics_d)
        wandb.log(self.summarizer.summarize())
        self.print_stats()

        # save final model
        if config["save"]:
            filename = "agent_final.pt"
            checkpoint = {"config": dict(config), "state_dict": self.agent.state_dict()}
            torch.save(checkpoint, os.path.join(wandb.run.dir, filename))
            wandb.save(filename)

    def _pretrain(self):
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        obs_model = wm.obs_model
        ac = agent.ac
        replay_buffer = self.replay_buffer

        logger.info("pretrain observation model")
        wm_total_batch_size = config["wm_batch_size"] * config["wm_sequence_length"]
        budget = config["pretrain_budget"] * config["pretrain_obs_p"]
        with tqdm(total=int(budget), mininterval=mininterval) as pbar:
            while budget > 0:
                indices = torch.randperm(
                    replay_buffer.size, device=replay_buffer.device
                )
                while len(indices) > 0 and budget > 0:
                    idx = indices[:wm_total_batch_size]
                    indices = indices[wm_total_batch_size:]
                    o = replay_buffer.get_obs(idx, device=device)
                    _ = wm.optimize_pretrain_obs(o.unsqueeze(1))
                    pbar.update(int(idx.numel()))
                    budget -= idx.numel()

        # encode all observations once, since the encoder does not change anymore
        indices = torch.arange(
            replay_buffer.size, dtype=torch.long, device=replay_buffer.device
        )
        o = replay_buffer.get_obs(
            indices.unsqueeze(0), prefix=1, device=device, return_next=True
        )  # 1 for context
        o = o.squeeze(0).unsqueeze(1)
        with torch.no_grad():
            z_dist = obs_model.eval().encode(o)

        logger.info("pretrain dynamics model")
        budget = config["pretrain_budget"] * config["pretrain_dyn_p"]
        with tqdm(total=int(budget), mininterval=mininterval) as pbar:
            while budget > 0:
                for idx in replay_buffer.generate_uniform_indices(
                    config["wm_batch_size"], config["wm_sequence_length"], extra=2
                ):  # 2 for context + next
                    z, logits = obs_model.sample_z(
                        z_dist, idx=idx.flatten(), return_logits=True
                    )
                    z, logits = [
                        x.squeeze(1).unflatten(0, idx.shape) for x in (z, logits)
                    ]
                    z = z[:, :-1]
                    target_logits = logits[:, 2:]
                    idx = idx[:, :-2]
                    _, a, r, terminated, truncated, _ = replay_buffer.get_data(
                        idx, device=device, prefix=1
                    )
                    _ = wm.optimize_pretrain_dyn(
                        z, a, r, terminated, truncated, target_logits
                    )
                    pbar.update(int(idx.numel()))
                    budget -= idx.numel()
                    if budget <= 0:
                        break

        logger.info("pretrain ac")
        budget = config["pretrain_budget"] * (
            1 - config["pretrain_obs_p"] + config["pretrain_dyn_p"]
        )
        with tqdm(total=budget * 1, mininterval=mininterval) as pbar:
            while budget > 0:
                for idx in replay_buffer.generate_uniform_indices(
                    config["ac_batch_size"], config["ac_horizon"], extra=2
                ):  # 2 for context + next
                    z = obs_model.sample_z(z_dist, idx=idx.flatten())
                    z = z.squeeze(1).unflatten(0, idx.shape)
                    idx = idx[:, :-2]
                    _, a, r, terminated, truncated, _ = replay_buffer.get_data(
                        idx, device=device, prefix=1
                    )
                    d = torch.logical_or(terminated, truncated)
                    if config["ac_input_h"]:
                        g = wm.to_discounts(terminated)
                        tgt_length = config["ac_horizon"] + 1
                        with torch.no_grad():
                            _, h, _ = wm.dyn_model.eval().predict(
                                z[:, :-1], a, r, g, d[:, :-1], tgt_length
                            )
                    else:
                        h = None
                    g = wm.to_discounts(d)
                    z, r, g, d = [x[:, 1:] for x in (z, r, g, d)]
                    _ = ac.optimize_pretrain(z, h, r, g, d)
                    budget -= idx.numel()
                    pbar.update(idx.numel() * 1.0)
                    if budget <= 0:
                        break
        ac.sync_target()

    def _train_step(self):
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        ac = agent.ac
        replay_buffer = self.replay_buffer

        # train wm
        for _ in range(config["wm_train_steps"]):
            metrics_i = {}
            idx = replay_buffer.sample_indices(
                config["wm_batch_size"], config["wm_sequence_length"]
            )
            o, a, r, terminated, truncated, _ = replay_buffer.get_data(
                idx, device=device, prefix=1, return_next_obs=True
            )  # 1 for context

            z, h, met = wm.optimize(o, a, r, terminated, truncated)
            metrics.update_metrics(metrics_i, met, prefix="wm/")

            o, a, r, terminated, truncated = [
                x[:, :-1] for x in (o, a, r, terminated, truncated)
            ]

        metrics_d = metrics_i  # only use last metrics

        # train actor-critic
        create_start = lambda x, size: utils.windows(x, size).flatten(0, 1)
        start_z = create_start(z, 2)
        start_a = create_start(a, 1)
        start_r = create_start(r, 1)
        start_terminated = create_start(terminated, 1)
        start_truncated = create_start(truncated, 1)

        idx = utils.random_choice(
            start_z.shape[0], config["ac_batch_size"], device=start_z.device
        )
        start_z, start_a, start_r, start_terminated, start_truncated = [
            x[idx]
            for x in (start_z, start_a, start_r, start_terminated, start_truncated)
        ]

        dreamer = Dreamer(
            config,
            wm,
            mode="imagine",
            ac=ac,
            store_data=True,
            start_z_sampler=self._create_start_z_sampler(temperature=1),
        )
        dreamer.imagine_reset(
            start_z, start_a, start_r, start_terminated, start_truncated
        )
        for _ in range(config["ac_horizon"]):
            a = dreamer.act()
            dreamer.imagine_step(a)
        z, o, h, a, r, g, d, weights = dreamer.get_data()
        if config["wm_discount_threshold"] == 0:
            d = None  # save some computation, since all dones are False in this case
        ac_metrics = ac.optimize(z, h, a, r, g, d, weights)
        metrics.update_metrics(metrics_d, ac_metrics, prefix="ac/")

        return metrics_d

    @torch.no_grad()
    def _evaluate(self, is_final):
        start_time = time.time()
        config = self.config
        agent = self.agent
        device = next(agent.parameters()).device
        wm = agent.wm
        ac = agent.ac
        replay_buffer = self.replay_buffer

        metrics_d = {}
        metrics_d["buffer/visits"] = replay_buffer.visit_histogram()
        metrics_d["buffer/sample_probs"] = replay_buffer.sample_probs_histogram()

        recon_img, imagine_img = self._create_eval_images(is_final)
        metrics_d["eval/recons"] = wandb.Image(recon_img)
        if imagine_img is not None:
            metrics_d["eval/imagine"] = wandb.Image(imagine_img)

        # similar to evaluation proposed in https://arxiv.org/pdf/2007.05929.pdf (SPR) section 4.1
        num_episodes = (
            config["final_eval_episodes"] if is_final else config["eval_episodes"]
        )
        num_envs = max(
            min(num_episodes, int(multiprocessing.cpu_count() * config["cpu_p"])), 1
        )
        # logger.debug(f'num_envs: {num_envs}')
        env_fn = lambda: Trainer._create_env_from_config(config, eval=True)

        # FIXME: this multiproc wont work with craftax jax
        num_envs = 1  # FIXME
        eval_env = env_fn()  # create_vector_env(num_envs, env_fn())

        seed = (
            ((config["seed"] + 13) * 79 + 13) if config["seed"] is not None else None
        )
        start_obs, _ = eval_env.reset(seed=seed)
        # add in fake batch and tgt_len dimensions
        start_obs = start_obs.unsqueeze(0).unsqueeze(1).to(device)
        # start_obs = preprocess_atari_obs(start_obs, device).unsqueeze(1)

        dreamer = Dreamer(config, wm, mode="observe", ac=ac, store_data=False)
        dreamer.observe_reset_single(start_obs)

        scores = []
        current_score = 0
        finished = 0
        num_truncated = 0
        while len(scores) < num_episodes:
            a = dreamer.act()
            o, r, terminated, truncated, infos = eval_env.step(
                a.squeeze(1).squeeze(0).cpu().numpy()
            )
            current_score += r
            finished = truncated | terminated

            # o = preprocess_atari_obs(o, device)
            o = o.unsqueeze(0).unsqueeze(1).to(device)
            r = (
                torch.as_tensor(r, dtype=torch.float, device=device)
                .unsqueeze(0)
                .unsqueeze(1)
            )
            terminated = (
                torch.as_tensor(terminated, device=device).unsqueeze(0).unsqueeze(1)
            )
            truncated = (
                torch.as_tensor(truncated, device=device).unsqueeze(0).unsqueeze(1)
            )
            z, h, _, d, _ = dreamer.observe_step(a, o, r, terminated, truncated)

            if finished:
                scores.append(current_score)
                num_scores = len(scores)
                if num_scores >= num_episodes:
                    if num_scores > num_episodes:
                        scores = scores[:num_episodes]  # unbiased, just pick first
                    break
                current_score = 0
                finished = False
                if seed is not None:
                    seed = np.int64(seed + 13 + num_envs)
                start_o, _ = eval_env.reset(seed=seed, options={"force": True})
                start_o = start_o.unsqueeze(0).unsqueeze(1).to(device)
                # start_o = preprocess_atari_obs(start_o, device).unsqueeze(1)
                dreamer = Dreamer(config, wm, mode="observe", ac=ac, store_data=False)
                dreamer.observe_reset_single(start_o)
        eval_env.close()
        if num_truncated > 0:
            print(f"{num_truncated} episode(s) truncated")

        score_mean = np.mean(scores)
        score_metrics = {
            "score_mean": score_mean,
            "score_std": np.std(scores),
            "score_median": np.median(scores),
            "score_min": np.min(scores),
            "score_max": np.max(scores),
            # 'hns': compute_atari_hns(config['game'], score_mean)
        }
        metrics_d.update({f"eval/{key}": value for key, value in score_metrics.items()})
        if is_final:
            metrics_d.update(
                {f"eval/final_{key}": value for key, value in score_metrics.items()}
            )

        end_time = time.time()
        eval_time = end_time - start_time

        self.total_eval_time += eval_time
        metrics_d["eval/total_time"] = self.total_eval_time

        self.last_eval = replay_buffer.size
        return metrics_d

    @torch.no_grad()
    def _create_eval_images(self, is_final=False):
        """Create images to QC reconstructed images and imagined images."""
        config = self.config
        agent = self.agent
        replay_buffer = self.replay_buffer
        device = next(agent.parameters()).device
        obs_model = agent.wm.obs_model.eval()

        # recon_img
        idx = utils.random_choice(
            replay_buffer.size, 10, device=replay_buffer.device
        ).unsqueeze(1)
        o = replay_buffer.get_obs(idx, device=device)

        z = obs_model.encode_sample(o, temperature=0)
        recons = obs_model.decode(z)
        # use last frame of frame stack [b=10, 1?, framestack=4, odim=8268[
        o = o[:, :, -1]
        recons = recons[:, :, -1]

        # for craftax convert state to image
        o = craftax_symobs_to_img(o.squeeze(1), self.env.unwrapped.env_state)
        recons = craftax_symobs_to_img(recons.squeeze(1), self.env.unwrapped.env_state)
        # render observations and reconstructions as two columns
        # the first is the original observation, the second is the reconstruction
        # QC: the reconstruction should look similar
        # recon_img = [o.permute(0, 3, 1, 2), recons.permute(0, 3, 1, 2)]
        # recon_img = torch.cat([o, recons], dim=0) / 255.0
        recon_img = rearrange([o, recons], 't b h w c -> (t b) c h w') / 255.0

        # FIXME:
        utils.to_image(torchvision.utils.make_grid(recon_img, nrow=o.shape[0], padding=2)).save('recon_img1.png')
        utils.to_image(torchvision.utils.make_grid(recon_img, nrow=o.shape[1], padding=2)).save('recon_img2.png')

        recon_img = torchvision.utils.make_grid(recon_img, nrow=o.shape[0], padding=2)
        recon_img = utils.to_image(recon_img)

        # imagine_img
        idx = idx[:5]
        start_o = replay_buffer.get_obs(idx, prefix=1, device=device)  # 1 for context
        start_a = replay_buffer.get_actions(idx, prefix=1, device=device)[:, :-1]
        start_r = replay_buffer.get_rewards(idx, prefix=1, device=device)[:, :-1]
        start_terminated = replay_buffer.get_terminated(idx, prefix=1, device=device)[
            :, :-1
        ]
        start_truncated = replay_buffer.get_truncated(idx, prefix=1, device=device)[
            :, :-1
        ]
        start_z = obs_model.encode_sample(start_o, temperature=0)

        horizon = 100 if is_final else config["wm_sequence_length"]
        dreamer = Dreamer(
            config,
            agent.wm,
            mode="imagine",
            ac=agent.ac,
            store_data=True,
            start_z_sampler=self._create_start_z_sampler(temperature=0),
            always_compute_obs=True,
        )
        dreamer.imagine_reset(
            start_z,
            start_a,
            start_r,
            start_terminated,
            start_truncated,
            keep_start_data=True,
        )
        for _ in range(horizon):
            a = dreamer.act()
            dreamer.imagine_step(a, temperature=1)
        z, o, _, a, r, g, d, weights = dreamer.get_data()

        o = o[:, :-1, -1:]  # remove last time step and use last frame of frame stack
        a, r, g, weights = [x.cpu().numpy() for x in (a, r, g, weights)]

        # make an imagined image where the rows are the rollout, cols are batch
        # that means you should look at each col and check for consistency and realism (are trees and coastlines teleporting? are the colors consistent? or is psychadelic?)
        # note that we fill in some values from the true state, so it's not a perfect representation
        # [b=5, t=17, 1, 8268] -> [5, 17, 130, 110, 3]
        pad = 2
        extra_pad = 38
        imagine_img2 = (
            craftax_symobs_to_img(o, self.env.unwrapped.env_state)
            .squeeze(2)
        )

        # FIXME: an attempt
        imagine_img = rearrange(imagine_img2, 'b t h w c -> (b t) c h w') / 255.0
        utils.to_image(utils.make_grid(imagine_img, nrow=o.shape[0], padding=(pad + extra_pad, pad))).save('imagine_img3.png')
        utils.to_image(utils.make_grid(imagine_img, nrow=o.shape[1], padding=(pad + extra_pad, pad))).save('imagine_img4.png')

        imagine_img = rearrange(imagine_img2, 'b t h w c -> (t b) c h w') / 255.0
        h, w = imagine_img.shape[-2:]

        # FIXME:
        utils.to_image(utils.make_grid(imagine_img, nrow=o.shape[0], padding=(pad + extra_pad, pad))).save('imagine_img1.png')
        utils.to_image(utils.make_grid(imagine_img, nrow=o.shape[1], padding=(pad + extra_pad, pad))).save('imagine_img2.png')


        imagine_img = utils.make_grid(
            imagine_img, nrow=o.shape[0], padding=(pad + extra_pad, pad)
        )
        imagine_img = utils.to_image(imagine_img[:, extra_pad:])

        # DELETEME
        # # r [b=5, t=17]
        # t,i = r.shape
        # t-=1
        # i-=1
        # y = pad + t * (w + pad)
        # x = pad + i * (h+extra_pad+pad)+h
        # print(f'image {np.array(imagine_img).shape}, {x},{y}')


        # draw action, reward, and discount on the imagined image for each batch
        draw = ImageDraw.Draw(imagine_img)
        for t in range(r.shape[1]): # timestep
            for i in range(r.shape[0]): # batch
                x = pad + t * (w + pad)
                y = pad + i * (h + extra_pad + pad) + h
                weight = weights[i, t]
                reward = r[i, t]

                if abs(reward) < 1e-4:
                    color_rgb = int(weight * 255)
                    color = (color_rgb, color_rgb, color_rgb)  # white
                elif reward > 0:
                    color_rb = int(weight * 100)
                    color_g = int(weight * (255 + reward * 255) / 2)
                    color = (color_rb, color_g, color_rb)  # green
                else:
                    color_gb = int(weight * 80)
                    color_r = int(weight * (255 + (-reward) * 255) / 2)
                    color = (color_r, color_gb, color_gb)  # red
                draw.text(
                    (x + 2, y + 2),
                    f"a: {self.action_meanings[a[i, t]][:7]: >7}",
                    fill=color,
                )
                draw.text((x + 2, y + 2 + 10), f"r: {r[i, t]: .4f}", fill=color)
                draw.text((x + 2, y + 2 + 20), f"g: {g[i, t]: .4f}", fill=color)
        return recon_img, imagine_img
