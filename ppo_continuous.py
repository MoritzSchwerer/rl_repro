import math
import numpy as np
import gymnasium as gym
import tyro
import torch
import torch.nn as nn
import time

from datetime import datetime
from dataclasses import dataclass
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from utils import ReplayBuffer


@dataclass
class PPOAgentArgs:
    activation: str = "nn.Tanh"
    """The activation to use in between linear layers"""
    initialize_orthogonal: bool = True
    """Weather to initialize the linear layers with orthogonal matricies"""
    initial_logstd: float = 0
    """The intial log standard deviation of the actor"""
    hidden_size: int = 256
    """The number of neurons in the hidden layers"""


@dataclass
class PPOArgs:
    """This script will run the PPO algorithm with the provided arguments
    and the default arguments."""

    experiment_name: str
    """The name of the experiment for saving models and logs"""
    env_id: str = "Humanoid-v5"
    """The environment to instantiate"""
    device: str = "cpu"
    """The device to run the neural nets on"""
    agent: PPOAgentArgs = PPOAgentArgs()

    lr: float = 3e-5
    """The learning rate of the actor"""
    critic_loss_coef: float = 0.5
    """~ Critic lr / actor lr"""
    entropy_loss_coef: float = 0.002
    """Coefficient of entropy loss"""
    clip_coef: float = 0.3
    """Probability ratio will be clipped to the range [1 - clip_coef, 1 + clip_coef]"""
    clip_value_loss: bool = True
    """If we also clip the value with clip_coef"""
    normalize_advantages: bool = True
    """If we normalize the advantages per minibatch"""
    anneal_lr: bool = True
    """If the learning rate will be annealed linearly towards 0"""
    gamma: float = 0.95
    """The discount factor"""
    lambda_: float = 0.90
    """The dropoff coefficient"""
    num_epochs: int = 5
    """How often the model iterates over the entire rollout buffer"""
    num_iter: int = 4000 * 7
    """The total number of repetitions of collecting data followed by training"""
    num_envs: int = 1
    """The number of parallel environments to run (usually 1)"""
    batch_size: int = 256
    """The batch size for training"""
    time_horizon: int = 512
    """The amount of steps to perform collecting data for before training"""
    max_grad_norm: float = 2
    """The maximum gradient norm, that won't be clipped"""

    @property
    def rollout_size(self) -> int:
        return self.time_horizon  # * self.num_envs


DEFAULT_CONFIGS = {
    "Hopper-v5-base": (
        "Default configuration for Hopper-v5 environment.",
        PPOArgs(
            experiment_name="Hopper-v5-base",
            env_id="Hopper-v5",
            device="cpu",
            agent=PPOAgentArgs(
                activation="nn.Tanh",
                initialize_orthogonal=True,
                initial_logstd=0,
                hidden_size=64,
            ),
            lr=3e-4,
            critic_loss_coef=0.5,
            entropy_loss_coef=0.000,
            clip_coef=0.2,
            clip_value_loss=True,
            normalize_advantages=True,
            anneal_lr=True,
            gamma=0.99,
            lambda_=0.95,
            num_epochs=10,
            num_iter=500,
            num_envs=1,
            batch_size=64,
            time_horizon=2048,
            max_grad_norm=2,
        ),
    ),
    "Humanoid-v5-base": (
        "Default configuration for Humanoid-v5 environment.",
        PPOArgs(
            experiment_name="Humanoid-v5-base",
            env_id="Humanoid-v5",
            device="cpu",
            agent=PPOAgentArgs(
                activation="nn.Tanh",
                initialize_orthogonal=True,
                initial_logstd=-2,
                hidden_size=256,
            ),
            lr=3e-5,
            critic_loss_coef=0.5,
            entropy_loss_coef=0.002,
            clip_coef=0.3,
            clip_value_loss=True,
            normalize_advantages=True,
            anneal_lr=True,
            gamma=0.95,
            lambda_=0.90,
            num_epochs=5,
            num_iter=28000,
            num_envs=1,
            batch_size=256,
            time_horizon=512,
            max_grad_norm=2,
        ),
    ),
}


def make_env(env_id: str, render_video: bool = False):
    if render_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{id}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.clip(obs, -10, 10), env.observation_space
    )
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def train(env, args):
    agent = Agent(env, args.agent).to(args.device)

    optim = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obs_dim = math.prod(env.observation_space.shape)
    act_dim = math.prod(env.action_space.shape)

    logger = SummaryWriter(
        log_dir=f"tb_logs/{args.experiment_name}_{datetime.now().strftime('%m_%d-%H:%M')}"
    )

    global_step = 0
    start_time = time.time()
    for it in range(args.num_iter):
        if args.anneal_lr:
            frac = 1.0 - it / args.num_iter
            optim.param_groups[0]["lr"] = args.lr * frac

        observations_t = torch.empty(
            (args.time_horizon, args.num_envs, obs_dim), device=args.device
        )
        rewards_t = torch.empty((args.time_horizon, args.num_envs), device=args.device)
        actions_t = torch.empty(
            (args.time_horizon, args.num_envs, act_dim), device=args.device
        )
        dones_t = torch.empty(
            (args.time_horizon, args.num_envs),
            device=args.device,
        )
        advantages_t = torch.empty(
            (args.time_horizon, args.num_envs), device=args.device
        )
        values_t = torch.empty((args.time_horizon, args.num_envs), device=args.device)
        log_probs_t = torch.empty(
            (args.time_horizon, args.num_envs), device=args.device
        )

        next_obs, info = env.reset()
        next_obs = torch.from_numpy(next_obs).to(args.device)
        next_dones = torch.empty(args.num_envs, device=args.device)

        # collect observations
        for t in range(args.time_horizon):
            global_step += args.num_envs
            observations_t[t] = next_obs.unsqueeze(0)
            dones_t[t] = next_dones.unsqueeze(0)

            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(next_obs)

            actions_t[t] = actions.unsqueeze(0)
            values_t[t] = values.unsqueeze(0)
            log_probs_t[t] = log_probs.unsqueeze(0)

            actions = actions.squeeze(0).cpu().numpy()
            next_obs, rewards, truncated, terminated, info = env.step([actions])
            next_obs, rewards, terminated, truncated = (
                torch.from_numpy(next_obs).to(args.device),
                torch.from_numpy(np.array(rewards)).to(args.device),
                torch.from_numpy(np.array(terminated)).to(args.device),
                torch.from_numpy(np.array(truncated)).to(args.device),
            )

            next_dones = torch.logical_or(terminated, truncated).float()

            rewards_t[t] = rewards.unsqueeze(0)

            if "episode" in info.keys():
                print(f"Step: {global_step}, Episodic_return: {info['episode']['r']}")
                logger.add_scalar(
                    "Stats/episodic_return", info["episode"]["r"], global_step
                )
                logger.add_scalar(
                    "Stats/episodic_length", info["episode"]["l"], global_step
                )
        future_values = agent.get_value(next_obs)

        # cumpute gae
        with torch.no_grad():
            gae = 0
            for t in reversed(range(args.time_horizon)):
                if t == args.time_horizon - 1:
                    next_non_terminal = 1.0 - next_dones
                    next_values = future_values
                else:
                    next_non_terminal = 1.0 - dones_t[t + 1]
                    next_values = values_t[t + 1]
                delta = (
                    rewards_t[t]
                    + args.gamma * next_values * next_non_terminal
                    - values_t[t]
                )
                gae = delta + args.gamma * args.lambda_ * next_non_terminal * gae
                advantages_t[t] = gae
            returns_t = advantages_t + values_t

        observations_t = observations_t.flatten(0, 1)
        actions_t = actions_t.flatten(0, 1)
        advantages_t = advantages_t.flatten()
        returns_t = returns_t.flatten()
        values_t = values_t.flatten()
        log_probs_t = log_probs_t.flatten()

        clip_fracs = []
        buffer_size = args.time_horizon * args.num_envs
        for step in range(args.num_epochs):
            indices = torch.randperm(buffer_size)
            for start_index in range(0, buffer_size, args.batch_size):
                end_index = min(start_index + args.batch_size, buffer_size)
                batch_inds = indices[start_index:end_index]

                observations = observations_t[batch_inds]
                actions = actions_t[batch_inds]
                advantages = advantages_t[batch_inds]
                returns = returns_t[batch_inds]
                old_values = values_t[batch_inds]
                old_logprobs = log_probs_t[batch_inds]

                if args.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                _, new_logprob, entropy, values_pred = agent.get_action_and_value(
                    observations, actions
                )

                # policy loss
                log_ratio = new_logprob - old_logprobs
                ratio = log_ratio.exp()
                policy_loss1 = -advantages * ratio
                policy_loss2 = -advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                # critic training
                values_pred = values_pred.flatten()
                if args.clip_value_loss:
                    v_loss_unclipped = torch.square(values_pred - returns)
                    v_clipped = old_values + torch.clamp(
                        values_pred - old_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = torch.square(v_clipped - returns)
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    critic_loss = 0.5 * v_loss_max.mean()
                else:
                    critic_loss = 0.5 * torch.square(values_pred - returns).mean()

                # actor learning
                entropy_loss = entropy.mean()
                total_loss = (
                    policy_loss
                    + args.critic_loss_coef * critic_loss
                    - args.entropy_loss_coef * entropy_loss
                )

                optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optim.step()

        # y_pred, y_true = values_t, returns_t
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar(
            "Stats/learning_rate", optim.param_groups[0]["lr"], global_step
        )
        logger.add_scalar("Loss/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("Loss/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("Loss/clipfrac", np.mean(clip_fracs), global_step)
        logger.add_scalar("Loss/value_loss", critic_loss.item(), global_step)
        logger.add_scalar("Loss/actor_loss", policy_loss.item(), global_step)
        logger.add_scalar("Loss/entropy", entropy_loss.item(), global_step)
        # logger.add_scalar("Loss/explained_variance", explained_var, global_step)
        print("SPS: ", int(global_step / (time.time() - start_time)))
        logger.add_scalar(
            "Stats/StepsPerSecond",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
    logger.close()


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        args,
    ):
        super(Agent, self).__init__()
        self.args = args
        self.critic = nn.Sequential(
            self.layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    self.args.hidden_size,
                ),
                use=self.args.initialize_orthogonal,
            ),
            eval(self.args.activation)(),
            self.layer_init(
                nn.Linear(self.args.hidden_size, self.args.hidden_size),
                use=self.args.initialize_orthogonal,
            ),
            eval(self.args.activation)(),
            self.layer_init(
                nn.Linear(self.args.hidden_size, 1),
                std=1.0,
                use=self.args.initialize_orthogonal,
            ),
        )
        self.actor_mean = nn.Sequential(
            self.layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    self.args.hidden_size,
                ),
                use=self.args.initialize_orthogonal,
            ),
            eval(self.args.activation)(),
            self.layer_init(
                nn.Linear(self.args.hidden_size, self.args.hidden_size),
                use=self.args.initialize_orthogonal,
            ),
            eval(self.args.activation)(),
            self.layer_init(
                nn.Linear(
                    self.args.hidden_size, np.prod(envs.single_action_space.shape)
                ),
                std=0.01,
                use=self.args.initialize_orthogonal,
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.full(
                (1, np.prod(envs.single_action_space.shape)),
                self.args.initial_logstd,
                dtype=torch.float32,
            )
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        log_prob = log_prob.sum(-1)
        entropy = probs.entropy().sum(-1)
        value = self.critic(x)
        return action, log_prob, entropy, value

    def layer_init(self, layer, std=math.sqrt(2), bias_const=0.0, use: bool = True):
        if use:
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(DEFAULT_CONFIGS)
    print(config)

    env = make_env(config.env_id)
    env = gym.vector.SyncVectorEnv([lambda: env])

    train(env, config)

    env.close()
    # model_path = Path(f"models/{id}/ppo.pth")
    # model_path.mkdir(parents=True, exist_ok=True)
    # torch.save(model.agent.state_dict(), model_path)
