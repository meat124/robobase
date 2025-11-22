import shutil
import signal
import sys
import time
import random
from typing import Callable, Any
from functools import partial
import logging

from gymnasium import spaces
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from robobase import utils
from robobase.envs.env import EnvFactory
from robobase.logger import Logger
from robobase.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from robobase.replay_buffer.replay_buffer import ReplayBuffer
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer


from pathlib import Path

import hydra
import numpy as np
import torch
import gymnasium as gym
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(int(seed))


def _create_default_replay_buffer(
    cfg: DictConfig,
    observation_space: gym.Space,
    action_space: gym.Space,
    demo_replay: bool = False,
) -> ReplayBuffer:
    extra_replay_elements = spaces.Dict({})
    if cfg.demos != 0:
        extra_replay_elements["demo"] = spaces.Box(0, 1, shape=(), dtype=np.uint8)
    # Create replay_class with buffer-specific hyperparameters
    replay_class = UniformReplayBuffer
    if cfg.replay.prioritization:
        replay_class = PrioritizedReplayBuffer
    replay_class = partial(
        replay_class,
        nstep=cfg.replay.nstep,
        gamma=cfg.replay.gamma,
    )
    # Create replay_class with common hyperparameters
    return replay_class(
        save_dir=cfg.replay.save_dir,
        batch_size=cfg.batch_size if not demo_replay else cfg.demo_batch_size,
        replay_capacity=cfg.replay.size if not demo_replay else cfg.replay.demo_size,
        action_shape=action_space.shape,
        action_dtype=action_space.dtype,
        reward_shape=(),
        reward_dtype=np.float32,
        observation_elements=observation_space,
        extra_replay_elements=extra_replay_elements,
        num_workers=cfg.replay.num_workers,
        sequential=cfg.replay.sequential,
    )


def _create_default_envs(cfg: DictConfig) -> EnvFactory:
    factory = None
    if cfg.env.env_name == "rlbench":
        from robobase.envs.rlbench import RLBenchEnvFactory

        factory = RLBenchEnvFactory()
    elif cfg.env.env_name == "dmc":
        from robobase.envs.dmc import DMCEnvFactory

        factory = DMCEnvFactory()
    elif cfg.env.env_name == "bigym":
        from robobase.envs.bigym import BiGymEnvFactory

        factory = BiGymEnvFactory()
    elif cfg.env.env_name == "bigym_hdf5":
        from robobase.envs.bigym_hdf5 import BiGymHDF5EnvFactory

        factory = BiGymHDF5EnvFactory()
    elif cfg.env.env_name == "d4rl":
        from robobase.envs.d4rl import D4RLEnvFactory

        factory = D4RLEnvFactory()
    else:
        ValueError()
    return factory


class Workspace:
    def __init__(
        self,
        cfg: DictConfig,
        env_factory: EnvFactory = None,
        create_replay_fn: Callable[[DictConfig], ReplayBuffer] = None,
        work_dir: str = None,
    ):
        print(f"{time.time():.2f}: Workspace __init__ started")
        if env_factory is None:
            env_factory = _create_default_envs(cfg)
        if create_replay_fn is None:
            create_replay_fn = _create_default_replay_buffer

        self.work_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            if work_dir is None
            else work_dir
        )
        print(f"workspace: {self.work_dir}")

        # Sanity checks
        if (
            not cfg.is_imitation_learning
            and cfg.replay_size_before_train * cfg.action_repeat * cfg.action_sequence
            < cfg.env.episode_length
            and cfg.replay_size_before_train > 0
        ):
            raise ValueError(
                "replay_size_before_train * action_repeat "
                f"({cfg.replay_size_before_train} * {cfg.action_repeat}) "
                f"must be >= episode_length ({cfg.env.episode_length})."
            )

        if cfg.method.is_rl and cfg.action_sequence != 1:
            raise ValueError("Action sequence > 1 is not supported for RL methods")
        if cfg.method.is_rl and cfg.execution_length != 1:
            raise ValueError("execution_length > 1 is not supported for RL methods")
        if not cfg.method.is_rl and cfg.replay.nstep != 1:
            raise ValueError("replay.nstep != 1 is not supported for IL methods")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        dev = "cpu"
        if cfg.num_gpus > 0:
            if sys.platform == "darwin":
                dev = "mps"
            else:
                dev = 0
                job_num = False
                try:
                    job_num = HydraConfig.get().job.get("num", False)
                except ValueError:
                    pass
                if job_num:
                    dev = job_num % cfg.num_gpus
        self.device = torch.device(dev)

        # create logger
        self.logger = Logger(self.work_dir, cfg=self.cfg)
        self.env_factory = env_factory

        if (num_demos := cfg.demos) != 0:
            # Collect demos or fetch saved demos before making environments
            # to consider demo-based action space (e.g., standardization)
            self.env_factory.collect_or_fetch_demos(cfg, num_demos)

        # Make training environment
        if cfg.num_train_envs > 0:
            self.train_envs = self.env_factory.make_train_env(cfg)
        else:
            self.train_envs = None
            logging.warning("Train env is not created. Training will not be supported ")

        # Create evaluation environment
        self.eval_env = self.env_factory.make_eval_env(cfg)

        if num_demos != 0:
            # Post-process demos using the information from environments
            self.env_factory.post_collect_or_fetch_demos(cfg)

        # Create the RL Agent
        observation_space = self.eval_env.observation_space
        action_space = self.eval_env.action_space

        intrinsic_reward_module = None
        if cfg.get("intrinsic_reward_module", None):
            intrinsic_reward_module = hydra.utils.instantiate(
                cfg.intrinsic_reward_module,
                device=self.device,
                observation_space=observation_space,
                action_space=action_space,
            )

        self.agent = hydra.utils.instantiate(
            cfg.method,
            device=self.device,
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=cfg.num_train_envs,
            replay_alpha=cfg.replay.alpha,
            replay_beta=cfg.replay.beta,
            frame_stack_on_channel=cfg.frame_stack_on_channel,
            intrinsic_reward_module=intrinsic_reward_module,
        )
        self.agent.train(False)

        self.replay_buffer = create_replay_fn(cfg, observation_space, action_space)
        self.prioritized_replay = cfg.replay.prioritization
        self.extra_replay_elements = self.replay_buffer.extra_replay_elements

        self.replay_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.batch_size,
            num_workers=cfg.replay.num_workers,
            pin_memory=cfg.replay.pin_memory,
            worker_init_fn=_worker_init_fn,
        )
        self._replay_iter = None

        # Create validation replay buffer for proper validation loss
        self.val_replay_buffer = None
        self.val_replay_loader = None
        self._val_replay_iter = None
        
        # Create a separate demo replay that contains successful episodes.
        # This is designed for RL. IL algorithms don't have to use this!
        # TODO: Change the name to `self_imitation_buffer` or other names
        # Note that original buffer also contains demos, but they are not protected
        # TODO: Support demo protection in a buffer
        self.use_demo_replay = cfg.demo_batch_size is not None
        if self.use_demo_replay:
            self.demo_replay_buffer = create_replay_fn(
                cfg, observation_space, action_space, demo_replay=True
            )
            self.demo_replay_loader = DataLoader(
                self.demo_replay_buffer,
                batch_size=self.demo_replay_buffer.batch_size,
                num_workers=cfg.replay.num_workers,
                pin_memory=cfg.replay.pin_memory,
                worker_init_fn=_worker_init_fn,
            )

        if self.prioritized_replay:
            if self.use_demo_replay:
                raise NotImplementedError(
                    "Demo replay is not compatible with prioritized replay"
                )

        # RLBench doesn't like it when we import cv2 before it, so moving
        # import here.
        from robobase.video import VideoRecorder

        self.eval_video_recorder = VideoRecorder(
            (self.work_dir / "eval_videos") if self.cfg.log_eval_video else None
        )

        self._timer = utils.Timer()
        self._pretrain_step = 0
        self._main_loop_iterations = 0
        self._global_env_episode = 0
        self._act_dim = self.eval_env.action_space.shape[0]
        if self.train_envs:
            self._episode_rollouts = [[] for _ in range(self.train_envs.num_envs)]
        else:
            self._episode_rollouts = []

        if cfg.num_eval_episodes == 0:
            # We no longer need the eval env
            self.eval_env.close()
            self.eval_env = None

        self._shutting_down = False
        print(f"{time.time():.2f}: Workspace __init__ finished")

    @property
    def pretrain_steps(self):
        return self._pretrain_step

    @property
    def main_loop_iterations(self):
        return self._main_loop_iterations

    @property
    def global_env_episodes(self):
        return self._global_env_episode

    @property
    def global_env_steps(self):
        """Total number of environment steps taken."""
        if not self.train_envs:
            # If train envs is not enabled, we are in pure evaluation mode.
            # Return 0 as there is no global frame.
            return 0

        # TODO: Pretrain_steps should not be included in env_steps, because it's
        # training steps but not environment steps. We need another PR to address this
        return (
            self._main_loop_iterations
            * self.cfg.action_repeat
            * self.train_envs.num_envs
            * self.cfg.action_sequence
            + self.pretrain_steps
        )

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            _replay_iter = iter(self.replay_loader)
            if self.use_demo_replay:
                _demo_replay_iter = iter(self.demo_replay_loader)
                _replay_iter = utils.merge_replay_demo_iter(
                    _replay_iter, _demo_replay_iter
                )
            self._replay_iter = _replay_iter
        return self._replay_iter

    def train(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        if not self.train_envs:
            raise Exception("Train envs not created! Train can't be called!")
        try:
            self._train()
        except Exception as e:
            self.shutdown()
            raise e

    def _train(self):
        # Load Demo
        print(f"{time.time():.2f}: Starting _load_demos...")
        self._load_demos()
        print(f"{time.time():.2f}: Finished _load_demos.")
        
        # Save training stats (action_stats, demo_info, etc.)
        if self.cfg.is_imitation_learning and self.cfg.demos > 0:
            print(f"{time.time():.2f}: Saving training stats...")
            self._save_training_stats()
            print(f"{time.time():.2f}: Finished saving training stats.")

        # Perform pretraining. This is suitable for behaviour cloning or Offline RL
        print(f"{time.time():.2f}: Starting _pretrain_on_demos...")
        self._pretrain_on_demos()
        print(f"{time.time():.2f}: Finished _pretrain_on_demos.")

        # Perform online rl with exploration.
        self._online_rl()

        if self.cfg.save_snapshot:
            self.save_snapshot()

        self.shutdown()

    def eval(self) -> dict[str, Any]:
        return self._eval(eval_record_all_episode=True)

    def _eval(self, eval_record_all_episode: bool = False) -> dict[str, Any]:
        if self.cfg.num_eval_episodes == 0:
            return {}

        # TODO: In future, this func could do with a further refactor
        self.agent.set_eval_env_running(True)
        step, episode, total_reward, successes = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        first_rollout = []
        metrics = {}
        while eval_until_episode(episode):
            observation, info = self.eval_env.reset()
            # eval agent always has last id (ids start from 0)
            self.agent.reset(self.main_loop_iterations, [self.train_envs.num_envs])
            enabled = eval_record_all_episode or episode == 0
            self.eval_video_recorder.init(self.eval_env, enabled=enabled)
            termination, truncation = False, False
            while not (termination or truncation):
                (
                    action,
                    (next_observation, reward, termination, truncation, next_info),
                    env_metrics,
                ) = self._perform_env_steps(observation, self.eval_env, True)
                observation = next_observation
                info = next_info
                metrics.update(env_metrics)
                # Below is testing a feature wich can be enforced in v6.
                # The ability will allow agent info to be passed to envirionments.
                # This will be habdy for rednering any auxiliary outputs.
                if "agent_act_info" in env_metrics:
                    if hasattr(self.eval_env, "give_agent_info"):
                        self.eval_env.give_agent_info(env_metrics["agent_act_info"])
                self.eval_video_recorder.record(self.eval_env)
                total_reward += reward
                step += 1
            if episode == 0:
                first_rollout = np.array(self.eval_video_recorder.frames)
            self.eval_video_recorder.save(f"{self.global_env_steps}.mp4")
            success = info.get("task_success")
            if success is not None:
                successes += np.array(success).astype(int).item()
            else:
                successes = None
            episode += 1
        metrics.update(
            {
                "episode_reward": total_reward / episode,
                "episode_length": step * self.cfg.action_repeat / episode,
            }
        )
        if successes is not None:
            metrics["episode_success"] = successes / episode
        if self.cfg.log_eval_video and len(first_rollout) > 0:
            metrics["eval_rollout"] = dict(video=first_rollout, fps=4)
        self.agent.set_eval_env_running(False)
        return metrics

    def _add_to_replay(
        self,
        actions,
        observations,
        rewards,
        terminations,
        truncations,
        infos,
        next_infos,
    ):
        # TODO: In future, this func could do with a further refactor
        # TODO: Add transitions into replay buffer in sliding window fashion??
        #      Currently, as train env has action sequence wrapper which only gives
        #      total reward and final obs for the full sequence, we can't perform
        #      sliding window.

        # Convert observation to list of observations ordered by train_env index
        list_of_obs_dicts = [
            dict(zip(observations, t)) for t in zip(*observations.values())
        ]
        agents_reset = []
        for i in range(self.train_envs.num_envs):
            # Add transitions to episode rollout
            self._episode_rollouts[i].append(
                (
                    actions[i],
                    list_of_obs_dicts[i],
                    rewards[i],
                    terminations[i],
                    truncations[i],
                    {k: infos[k][i] for k in infos.keys()},
                    {k: next_infos[k][i] for k in next_infos.keys()},
                )
            )

            # If episode finishes, add to replay buffer.
            if terminations[i] or truncations[i]:
                agents_reset.append(i)
                ep = self._episode_rollouts[i]
                last_next_info = ep[-1][-1]
                assert last_next_info["_final_observation"]
                # `next_info` containing `final_info` is the first info of next episode
                # we need to extract `final_info` and use it as true next_info
                final_obs = last_next_info["final_observation"]
                final_info = last_next_info["final_info"]
                task_success = int(final_info.get("task_success", 0) > 0.0)

                # Re-labeling successful demonstrations as success, following CQN
                relabeling_as_demo = (
                    task_success
                    and self.use_demo_replay
                    and self.cfg.use_self_imitation
                )
                for act, obs, rew, term, trunc, info, next_info in ep:
                    # Only keep the last frames regardless of frame stacks because
                    # replay buffer always store single-step transitions
                    obs = {k: v[-1] for k, v in obs.items()}

                    # Strip out temporal dimension as action_sequence = 1
                    act = act[0]

                    if relabeling_as_demo:
                        info["demo"] = 1
                    else:
                        info["demo"] = 0

                    # Filter out unwanted keys in info
                    extra_replay_elements = {
                        k: v
                        for k, v in info.items()
                        if k in self.extra_replay_elements.keys()
                    }

                    self.replay_buffer.add(
                        obs, act, rew, term, trunc, **extra_replay_elements
                    )
                    if relabeling_as_demo:
                        self.demo_replay_buffer.add(
                            obs, act, rew, term, trunc, **extra_replay_elements
                        )

                # Add final obs
                # Only keep the last frames regardless of frame stacks because
                # replay buffer always store single-step transitions
                final_obs = {k: v[-1] for k, v in final_obs.items()}
                self.replay_buffer.add_final(final_obs)
                if relabeling_as_demo:
                    self.demo_replay_buffer.add_final(final_obs)

                # clean up
                self._global_env_episode += 1
                self._episode_rollouts[i].clear()

        self.agent.reset(self.main_loop_iterations, agents_reset)  # clear hidden dim

    def _signal_handler(self, sig, frame):
        print("\nCtrl+C detected. Preparing to shutdown...")
        self._shutting_down = True

    def _load_demos(self):
        if (num_demos := self.cfg.demos) != 0:
            # NOTE: Currently we do not protect demos from being evicted from replay
            self.env_factory.load_demos_into_replay(
                self.cfg,
                self.replay_buffer,
                is_demo_buffer=True if self.cfg.is_imitation_learning else False,
            )
            
            # Load validation demos into separate buffer
            if hasattr(self.env_factory, 'get_val_demos') and hasattr(self.env_factory, 'load_val_demos_into_replay'):
                val_demos = self.env_factory.get_val_demos()
                if val_demos is not None and len(val_demos) > 0:
                    # Create validation replay buffer using the same function
                    self.val_replay_buffer = _create_default_replay_buffer(
                        self.cfg, 
                        self.eval_env.observation_space, 
                        self.eval_env.action_space
                    )
                    
                    # Load validation demos using the factory method
                    self.env_factory.load_val_demos_into_replay(
                        self.cfg,
                        self.val_replay_buffer,
                        is_demo_buffer=False
                    )
                    
                    # Create validation data loader
                    from torch.utils.data import DataLoader
                    self.val_replay_loader = DataLoader(
                        self.val_replay_buffer,
                        batch_size=self.replay_buffer.batch_size,
                        num_workers=self.cfg.replay.num_workers,
                        pin_memory=self.cfg.replay.pin_memory,
                        worker_init_fn=_worker_init_fn,
                    )
                    logging.info(f"Validation buffer created with {len(self.val_replay_buffer)} transitions")
            
            if self.use_demo_replay:
                # Load demos to the dedicated demo_replay_buffer
                self.env_factory.load_demos_into_replay(
                    self.cfg, self.demo_replay_buffer, is_demo_buffer=True
                )

        if self.cfg.replay_size_before_train > 0:
            diff = self.cfg.replay_size_before_train - len(self.replay_buffer)
            if num_demos > 0 and diff > 0:
                logging.warning(
                    f"Collecting additional {diff} random samples even though there "
                    f"are {len(self.replay_buffer)} demo samples inside the buffer. "
                    "Please make sure that this is an intended behavior."
                )

    def _perform_updates(self) -> dict[str, Any]:
        if self.agent.logging:
            start_time = time.time()
        metrics = dict()
        self.agent.train(True)
        for i in range(self.train_envs.num_envs):
            if (self.main_loop_iterations + i) % self.cfg.update_every_steps != 0:
                # Skip update
                continue
            
            metrics.update(
                self.agent.update(
                    self.replay_iter, self.main_loop_iterations + i, self.replay_buffer
                )
            )

        self.agent.train(False)
        if self.agent.logging:
            execution_time_for_update = time.time() - start_time
            metrics["agent_batched_updates_per_second"] = (
                self.train_envs.num_envs / execution_time_for_update
            )
            metrics["agent_updates_per_second"] = (
                self.train_envs.num_envs * self.cfg.batch_size
            ) / execution_time_for_update
        return metrics

    def _perform_env_steps(
        self, observations: dict[str, np.ndarray], env: gym.Env, eval_mode: bool
    ) -> tuple[np.ndarray, tuple, dict[str, Any]]:
        if self.agent.logging:
            start_time = time.time()
        with torch.no_grad(), utils.eval_mode(self.agent):
            torch_observations = {
                k: torch.from_numpy(v).to(self.device) for k, v in observations.items()
            }
            if eval_mode:
                torch_observations = {
                    k: v.unsqueeze(0) for k, v in torch_observations.items()
                }
            action = self.agent.act(
                torch_observations, self.main_loop_iterations, eval_mode=eval_mode
            )
            metrics = {}
            # Below is testing a feature which can be enforced in v6.
            # The ability will allow agent info to be passed to environments.
            # This will be handy for rendering any auxiliary outputs.
            if isinstance(action, tuple):
                action, act_info = action
                metrics["agent_act_info"] = act_info
            action = action.cpu().detach().numpy()
            if action.ndim != 3:
                raise ValueError(
                    "Expected actions from `agent.act` to have shape "
                    "(Batch, Timesteps, Action Dim)."
                )
            if eval_mode:
                action = action[0]  # we expect batch of 1 for eval

        if self.agent.logging:
            execution_time_for_act = time.time() - start_time
            metrics["agent_act_steps_per_second"] = (
                self.train_envs.num_envs / execution_time_for_act
            )
            start_time = time.time()

        *env_step_tuple, next_info = env.step(action)

        if self.agent.logging:
            execution_time_for_env_step = time.time() - start_time
            metrics["env_steps_per_second"] = (
                self.train_envs.num_envs / execution_time_for_env_step
            )
            for k, v in next_info.items():
                # if train env, then will be vectorised, so get first elem
                metrics[f"env_info/{k}"] = v if eval_mode else v[0]

        return action, (*env_step_tuple, next_info), metrics

    def _compute_val_loss(self, num_batches: int = 10) -> float:
        """
        Compute validation loss on held-out validation set.
        Uses validation replay buffer that contains separate demos.
        
        Args:
            num_batches: Number of batches to sample for validation
            
        Returns:
            Average validation loss
        """
        import torch
        
        if self.val_replay_buffer is None or len(self.val_replay_buffer) == 0:
            logging.warning("No validation buffer available, skipping val_loss computation")
            return 0.0
        
        self.agent.train(False)  # Set to eval mode
        val_losses = []
        
        # Create iterator if needed
        if self._val_replay_iter is None:
            self._val_replay_iter = iter(self.val_replay_loader)
        
        try:
            for _ in range(num_batches):
                try:
                    batch = next(self._val_replay_iter)
                except StopIteration:
                    # Reset iterator if we run out of data
                    self._val_replay_iter = iter(self.val_replay_loader)
                    batch = next(self._val_replay_iter)
                
                # Get observations and actions from batch
                # Batch contains flattened observation keys + action, reward, etc.
                # Need to reconstruct observation dictionary from batch
                action = batch['action'].to(self.device)
                
                # Reconstruct observation dictionary
                obs = {}
                for key, value in batch.items():
                    # Skip non-observation keys
                    if key in ['action', 'reward', 'terminal', 'truncated', 'indices', 'discount', 'demo']:
                        continue
                    # Skip next-step observations (_tp1)
                    if key.endswith('_tp1'):
                        continue
                    # Add observation to dict
                    if isinstance(value, torch.Tensor):
                        obs[key] = value.to(self.device)
                
                with torch.no_grad():
                    # Import utility functions from agent
                    from robobase.method.utils import (
                        extract_from_spec,
                        extract_many_from_spec,
                        flatten_time_dim_into_channel_dim,
                        stack_tensor_dictionary
                    )
                    
                    # Process observations same way as agent._act()
                    low_dim_obs = None
                    fused_rgb_feats = None
                    
                    if 'low_dim_state' in obs:
                        low_dim_obs = flatten_time_dim_into_channel_dim(obs['low_dim_state'])
                    
                    # Process RGB observations
                    rgb_keys = [k for k in obs.keys() if k.startswith('rgb')]
                    if rgb_keys and self.agent.encoder is not None:
                        rgb_dict = {k: obs[k] for k in sorted(rgb_keys)}
                        rgb_obs = flatten_time_dim_into_channel_dim(
                            stack_tensor_dictionary(rgb_dict, 1),
                            has_view_axis=True
                        )
                        multi_view_rgb_feats = self.agent.encoder(rgb_obs.float())
                        fused_rgb_feats = self.agent.view_fusion(multi_view_rgb_feats) if self.agent.view_fusion else multi_view_rgb_feats
                    
                    # Compute loss without backprop
                    if hasattr(self.agent, 'compute_actor_loss'):
                        loss = self.agent.compute_actor_loss(obs, action)
                    else:
                        # Use actor's forward pass for diffusion policy
                        import torch.nn.functional as F
                        noise_pred, noise = self.agent.actor(low_dim_obs, fused_rgb_feats, action)
                        loss = F.mse_loss(noise_pred, noise)
                    
                    val_losses.append(loss.item())
        except Exception as e:
            import traceback
            logging.warning(f"Error computing validation loss: {e}")
            logging.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            self.agent.train(True)  # Set back to train mode
        
        return sum(val_losses) / len(val_losses) if val_losses else 0.0

    def _pretrain_on_demos(self):
        if self.cfg.num_pretrain_steps > 0:
            from tqdm import tqdm
            
            pre_train_until_step = utils.Until(self.cfg.num_pretrain_steps)
            should_pretrain_log = utils.Every(self.cfg.log_pretrain_every)
            should_pretrain_eval = utils.Every(self.cfg.eval_every_steps)
            # Add validation loss computation schedule
            val_loss_every = getattr(self.cfg, 'val_loss_every', self.cfg.log_pretrain_every)
            should_compute_val_loss = utils.Every(val_loss_every)
            
            if self.cfg.log_pretrain_every > 0:
                assert self.cfg.num_pretrain_steps % self.cfg.log_pretrain_every == 0
            if len(self.replay_buffer) <= 0:
                raise ValueError(
                    "there is no sample to pre-train with in the replay buffer "
                    f"but num_pretrain_steps ({self.cfg.num_pretrain_steps}) is > 0"
                )

            # Initialize tqdm progress bar
            pbar = tqdm(
                total=self.cfg.num_pretrain_steps,
                desc="🚀 Pretraining",
                unit="step",
                ncols=140,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
            
            # Track metrics for tqdm display
            last_loss = None
            last_val_loss = None
            last_success_rate = None

            while pre_train_until_step(self.pretrain_steps):
                # Check for shutdown signal
                if self._shutting_down:
                    print("\n⚠️  Shutdown requested. Stopping pretraining...")
                    break
                
                self.agent.logging = False

                if should_pretrain_log(self.pretrain_steps):
                    self.agent.logging = True
                
                start_time = time.time()
                pretrain_metrics = self._perform_updates()
                elapsed_time = time.time() - start_time

                # Update tracked metrics
                if 'actor_loss' in pretrain_metrics:
                    last_loss = pretrain_metrics['actor_loss']

                # Update tqdm with metrics
                postfix_dict = {}
                if last_loss is not None:
                    postfix_dict['train_loss'] = f"{last_loss:.4f}"
                if last_val_loss is not None:
                    postfix_dict['val_loss'] = f"{last_val_loss:.4f}"
                if last_success_rate is not None:
                    postfix_dict['success'] = f"{last_success_rate:.2%}"
                postfix_dict['time/step'] = f"{elapsed_time:.3f}s"
                
                pbar.set_postfix(postfix_dict)
                pbar.update(1)

                if should_pretrain_log(self.pretrain_steps):
                    pretrain_metrics.update(self._get_common_metrics())
                    
                    # Compute validation loss only when scheduled
                    if should_compute_val_loss(self.pretrain_steps):
                        val_loss = self._compute_val_loss(num_batches=10)
                        pretrain_metrics['val_loss'] = val_loss
                        last_val_loss = val_loss  # Update for progress bar
                    
                    self.logger.log_metrics(
                        pretrain_metrics, self.pretrain_steps, prefix="pretrain"
                    )
                    
                    # Simplified progress update (less verbose than before)
                    # Only print summary every log interval
                    if self.pretrain_steps % (self.cfg.log_pretrain_every * 10) == 0:
                        print(f"\n{'='*80}")
                        print(f"📊 Step {self.pretrain_steps}/{self.cfg.num_pretrain_steps} - Training Progress:")
                        print(f"{'='*80}")
                        
                        # Show key metrics only
                        if 'actor_loss' in pretrain_metrics:
                            print(f"  Loss: {pretrain_metrics['actor_loss']:.6f}", end="")
                            if 'val_loss' in pretrain_metrics:
                                val_loss = pretrain_metrics['val_loss']
                                print(f" | Val Loss: {val_loss:.6f}", end="")
                                # Overfitting warning
                                if val_loss > pretrain_metrics['actor_loss'] * 1.5:
                                    print(" ⚠️ (possible overfit)", end="")
                            print()
                        
                        if last_success_rate is not None:
                            print(f"  Success Rate: {last_success_rate:.2%}")
                        
                        print(f"{'='*80}\n")

                # Skip evaluation at step 0
                if should_pretrain_eval(self.pretrain_steps) and self.pretrain_steps > 0:
                    print(f"\n{'='*80}")
                    print(f"🎯 Evaluating at step {self.pretrain_steps}...")
                    print(f"{'='*80}")
                    
                    eval_metrics = self._eval()
                    
                    # Update success rate for tqdm
                    if 'episode_success' in eval_metrics:
                        last_success_rate = eval_metrics['episode_success']
                    
                    eval_metrics.update(self._get_common_metrics())
                    self.logger.log_metrics(
                        eval_metrics, self.pretrain_steps, prefix="pretrain_eval"
                    )
                    
                    # Print compact evaluation results
                    print(f"\n✅ Evaluation Results:", end="")
                    
                    if 'episode_success' in eval_metrics:
                        success_rate = eval_metrics['episode_success']
                        emoji = "🎉" if success_rate > 0.8 else "📈" if success_rate > 0.5 else "📊"
                        print(f" {emoji} Success={success_rate:.1%}", end="")
                    
                    if 'episode_reward' in eval_metrics:
                        print(f" | Reward={eval_metrics['episode_reward']:.2f}", end="")
                    
                    if 'episode_length' in eval_metrics:
                        print(f" | Length={eval_metrics['episode_length']:.0f}", end="")
                    
                    print()
                    print(f"{'='*80}\n")


                self._pretrain_step += 1
            
            pbar.close()
            print(f"\n{'='*80}")
            print("✅ Pretraining completed!")
            print(f"{'='*80}")
            if last_loss is not None:
                print(f"Final Loss: {last_loss:.6f}")
            if last_success_rate is not None:
                print(f"Latest Success Rate: {last_success_rate:.2%}")
            print(f"Total Steps: {self.cfg.num_pretrain_steps}")
            print(f"{'='*80}\n")

    def _online_rl(self):
        train_until_frame = utils.Until(self.cfg.num_train_frames)
        seed_until_size = utils.Until(self.cfg.replay_size_before_train)
        should_log = utils.Every(self.cfg.log_every)
        eval_every_n = self.cfg.eval_every_steps if self.eval_env is not None else 0
        should_eval = utils.Every(eval_every_n)
        snapshot_every_n = self.cfg.snapshot_every_n if self.cfg.save_snapshot else 0
        should_save_snapshot = utils.Every(snapshot_every_n)
        observations, info = self.train_envs.reset()
        #  We use agent 0 to accumulate stats about how the training agents are doing
        agent_0_ep_len = agent_0_reward = 0
        agent_0_prev_ep_len = agent_0_prev_reward = None
        while train_until_frame(self.global_env_steps):
            metrics = {}
            self.agent.logging = False
            if should_log(self.main_loop_iterations):
                self.agent.logging = True
            if not seed_until_size(len(self.replay_buffer)):
                update_metrics = self._perform_updates()
                metrics.update(update_metrics)

            (
                action,
                (next_observations, rewards, terminations, truncations, next_info),
                env_metrics,
            ) = self._perform_env_steps(observations, self.train_envs, False)

            agent_0_reward += rewards[0]
            agent_0_ep_len += 1
            if terminations[0] or truncations[0]:
                agent_0_prev_ep_len = agent_0_ep_len
                agent_0_prev_reward = agent_0_reward
                agent_0_ep_len = agent_0_reward = 0

            metrics.update(env_metrics)
            self._add_to_replay(
                action,
                observations,
                rewards,
                terminations,
                truncations,
                info,
                next_info,
            )
            observations = next_observations
            info = next_info
            if should_log(self.main_loop_iterations):
                metrics.update(self._get_common_metrics())
                if agent_0_prev_reward is not None and agent_0_prev_ep_len is not None:
                    metrics.update(
                        {
                            "episode_reward": agent_0_prev_reward,
                            "episode_length": agent_0_prev_ep_len
                            * self.cfg.action_repeat,
                        }
                    )
                self.logger.log_metrics(metrics, self.global_env_steps, prefix="train")

            if should_eval(self.main_loop_iterations):
                eval_metrics = self._eval()
                eval_metrics.update(self._get_common_metrics())
                self.logger.log_metrics(
                    eval_metrics, self.global_env_steps, prefix="eval"
                )

            if should_save_snapshot(self.main_loop_iterations):
                self.save_snapshot()

            if self._shutting_down:
                break

            self._main_loop_iterations += 1

    def _get_common_metrics(self) -> dict[str, Any]:
        _, total_time = self._timer.reset()
        metrics = {
            "total_time": total_time,
            "iteration": self.main_loop_iterations,
            "env_steps": self.global_env_steps,
            "env_episodes": self.global_env_episodes,
            "buffer_size": len(self.replay_buffer),
        }
        if self.use_demo_replay:
            metrics["demo_buffer_size"] = len(self.demo_replay_buffer)
        return metrics

    def shutdown(self):
        if self.eval_env:
            self.eval_env.close()

        self.train_envs.close()
        self.replay_buffer.shutdown()
        if self.use_demo_replay:
            self.demo_replay_buffer.shutdown()

    def save_snapshot(self):
        # Use 'ckpt' directory for consistency with BRS training
        snapshot = self.work_dir / "ckpt" / f"{self.global_env_steps}_snapshot.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        keys_to_save = [
            "_pretrain_step",
            "_main_loop_iterations",
            "_global_env_episode",
            "cfg",
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload["agent"] = self.agent.state_dict()
        with snapshot.open("wb") as f:
            torch.save(payload, f)
        latest_snapshot = self.work_dir / "ckpt" / "latest_snapshot.pt"
        shutil.copy(snapshot, latest_snapshot)

    def load_snapshot(self, path_to_snapshot_to_load=None):
        if path_to_snapshot_to_load is None:
            path_to_snapshot_to_load = (
                self.work_dir / "ckpt" / "latest_snapshot.pt"
            )
        else:
            path_to_snapshot_to_load = Path(path_to_snapshot_to_load)
        if not path_to_snapshot_to_load.is_file():
            raise ValueError(
                f"Provided file '{str(path_to_snapshot_to_load)}' is not a snapshot."
            )
        with path_to_snapshot_to_load.open("rb") as f:
            payload = torch.load(f, map_location="cpu")
        self.agent.load_state_dict(payload.pop("agent"))
        for k, v in payload.items():
            self.__dict__[k] = v
    
    def _save_training_stats(self):
        """
        Save training statistics (action stats, demo info, etc.) to work_dir.
        This ensures reproducibility and allows loading the same normalization stats.
        """
        if not self.cfg.is_imitation_learning or self.cfg.demos == 0:
            return
        
        import json
        stats_dir = self.work_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Save action statistics if available
        if hasattr(self.env_factory, '_action_stats') and self.env_factory._action_stats is not None:
            action_stats = self.env_factory._action_stats
            # Convert numpy arrays to lists for JSON serialization
            action_stats_json = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in action_stats.items()
            }
            action_stats_path = stats_dir / "action_stats.json"
            with open(action_stats_path, 'w') as f:
                json.dump(action_stats_json, f, indent=2)
            logging.info(f"✓ Saved action_stats.json to {action_stats_path}")
        
        # Save observation statistics if available
        if self.cfg.norm_obs and hasattr(self.replay_buffer, '_obs_mean'):
            obs_stats = {
                "obs_mean": self.replay_buffer._obs_mean.tolist() if hasattr(self.replay_buffer, '_obs_mean') else None,
                "obs_std": self.replay_buffer._obs_std.tolist() if hasattr(self.replay_buffer, '_obs_std') else None,
            }
            obs_stats_path = stats_dir / "obs_stats.json"
            with open(obs_stats_path, 'w') as f:
                json.dump(obs_stats, f, indent=2)
            logging.info(f"✓ Saved obs_stats.json to {obs_stats_path}")
        
        logging.info(f"\n📊 Training stats saved to: {stats_dir}")
        logging.info(f"  - action_stats.json")
        logging.info(f"  - demo_info.json")
        if self.cfg.norm_obs:
            logging.info(f"  - obs_stats.json")
        print()
