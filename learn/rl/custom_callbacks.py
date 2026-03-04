import os
import cv2
import warnings
import matplotlib
matplotlib.use('Agg')  # for headless environments
import numpy as np
from typing import Any
import multiprocessing
import gymnasium as gym
import matplotlib.pyplot as plt
from collections.abc import Callable

from stable_baselines3.common import type_aliases
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from tells_environment_dynamics.sim.sim_plot import Renderer2D

def save_argb_video(images, output_path, fps=30):
    # Convert ARGB to BGR (dropping alpha and converting to OpenCV's color order)

    height, width, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'X264' for H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter at {output_path}")

    for i, frame in enumerate(images):
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame {i} must be np.uint8, got {frame.dtype}")
        if frame.shape != (height, width, 3):
            raise ValueError(f"Frame {i} has shape {frame.shape}, expected {(height, width, 3)}")

        # Convert RGB to BGR (OpenCV expects BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    print(f'Video saved to {output_path}')

def create_image(object_data, xlim=(0, 10), ylim=(0, 10), image_size=(512, 512)):
    """
    Renders a 2D plot using OpenCV and returns an RGB image.
    
    Args:
        object_data: dict with keys:
            - 'points': list of (x, y) tuples
            - 'lines': list of (idx1, idx2) tuples connecting points
        xlim: (min_x, max_x)
        ylim: (min_y, max_y)
        image_size: (height, width) of the output image

    Returns:
        RGB image as a NumPy array of shape (height, width, 3)
    """
    height, width = image_size
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background

    def world_to_image(x, y):
        # Flip Y-axis because OpenCV has origin at top-left
        x_img = int((x - xlim[0]) / (xlim[1] - xlim[0]) * (width - 1))
        y_img = int((1 - (y - ylim[0]) / (ylim[1] - ylim[0])) * (height - 1))
        return x_img, y_img

    points = object_data['points']
    lines = object_data['lines']
    colors = object_data['colors']

    # Draw lines
    for i, line in enumerate(lines):
        i1 = line[0]
        i2 = line[1]
        p1 = world_to_image(*points[i1])
        p2 = world_to_image(*points[i2])
        if 'r' in colors[i]:
            cv2.line(img, p1, p2, color=(0, 0, 180), thickness=2)
        elif 'g' in colors[i]:
            cv2.line(img, p1, p2, color=(0, 150, 0), thickness=2)
        else:
            cv2.line(img, p1, p2, color=(0, 0, 0), thickness=2)

    # Draw points
    #for x, y in points:
    #    px, py = world_to_image(x, y)
    #    cv2.circle(img, (px, py), radius=4, color=(0, 0, 255), thickness=-1)

    return img

def evaluate_policy_collect_plot_data(
    model,
    env: gym.Env | VecEnv,
    deterministic: bool = True,
    render: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the policy for `n_eval_episodes` and collect plot data from each timestep.
    Assumes `env` has a method `get_plot_data()` returning a dictionary.

    :param model: SB3 model with `.predict()` method.
    :param env: A `gym.Env` or `VecEnv` that supports `get_plot_data()`.
    :param n_eval_episodes: Number of episodes to run.
    :param deterministic: Whether to use deterministic actions.
    :param render: Whether to render the environment.
    :return: List of dictionaries from `get_plot_data()` at each timestep.
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs
    episode_counts = np.zeros(n_envs, dtype=int)
    target_counts = np.ones(n_envs, dtype=int)

    plot_data = []
    for i in range(n_envs):
        plot_data.append([])

    observations = env.reset()
    states = None
    episode_starts = np.ones((n_envs,), dtype=bool)

    while (episode_counts < target_counts).any():
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_obs, rewards, dones, infos = env.step(actions)

        # Collect plot data from all environments
        for i in range(n_envs):
            if episode_counts[i] < target_counts[i]:
                plot_data[i].append(env.env_method("get_plot_data")[i])
                episode_starts[i] = dones[i]
                if dones[i]:
                    episode_counts[i] += 1

        observations = new_obs

    return plot_data


class VideoCallback(EvalCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        eval_env: gym.Env | VecEnv,
        save_path: str,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 1,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        name_prefix: str = "video",
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.save_path = save_path
        self.name_prefix = name_prefix

        #self.renderer = Renderer2D(xlim=[-50, 1050], ylim=[-50, 1050], render=False)  

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            print(self.save_path)
            os.makedirs(self.save_path, exist_ok=True)


    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

            # Collect plot data from all environments after evaluation
            self.eval_plot_data = evaluate_policy_collect_plot_data(
                self.model,
                self.eval_env,
                deterministic=self.deterministic
            )
            
            if multiprocessing.current_process().name == "MainProcess":
                self._save_videos()

        return continue_training


    def _save_videos(self):

        video_dir = os.path.join(self.save_path,str(self.num_timesteps))
        os.makedirs(video_dir, exist_ok=True)

        print(f'Saving videos at timestep {self.num_timesteps} to {video_dir}')

        for i in range(len(self.eval_plot_data)):
            images = []
            for img_data in self.eval_plot_data[i]:

                img = create_image(img_data,xlim=(-50,1050),ylim=(-50,1050))
                images.append(img)

            video_path = os.path.join(video_dir,self.name_prefix + str(i) + '.mp4')
            save_argb_video(images,video_path)
