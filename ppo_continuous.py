import math
import numpy as np
import gymnasium as gym
import tyro
import torch
import torch.nn as nn
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
        return self.time_horizon * self.num_envs


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
    env = gym.make(env_id)  # , render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, f"videos/{id}")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.clip(obs, -10, 10), env.observation_space
    )
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


class PPO:
    def __init__(self, env, args):
        self.env = env
        self.args: PPOArgs = args

        self.agent = Agent(self.env, self.args.agent).to(self.args.device)
        self.optim = torch.optim.Adam(
            self.agent.parameters(), lr=self.args.lr, eps=1e-5
        )

        self.obs_dim = math.prod(env.observation_space.shape)
        self.act_dim = math.prod(env.action_space.shape)

        self.rollout_buffer = ReplayBuffer(
            self.args.rollout_size,
            self.args.num_envs,
            (self.obs_dim,),
            (self.act_dim,),
            gamma=self.args.gamma,
            lmda=self.args.lambda_,
            device=self.args.device,
        )
        now = datetime.now().strftime("%m_%d-%H:%M")
        self.logger = SummaryWriter(
            log_dir=f"tb_logs/{self.args.experiment_name}_{now}"
        )

    def train(self) -> None:
        self.global_step = 0
        for it in range(self.args.num_iter):
            if self.args.anneal_lr:
                frac = 1.0 - it / self.args.num_iter
                self.optim.param_groups[0]["lr"] = self.args.lr * frac

            _ = self.collect_data()
            assert (
                self.rollout_buffer.is_full()
            ), "RolloutBuffer should be full at this point"

            _ = self.update()
        self.logger.close()

    def collect_data(self):
        self.rollout_buffer.clear()
        # initialize the buffers for target calculations
        observations_t = torch.zeros(
            (self.args.time_horizon, self.args.num_envs, self.obs_dim),
            device=self.args.device,
        )
        rewards_t = torch.zeros(
            (self.args.time_horizon, self.args.num_envs), device=self.args.device
        )
        actions_t = torch.zeros(
            (self.args.time_horizon, self.args.num_envs, self.act_dim),
            device=self.args.device,
        )
        dones_t = torch.zeros(
            (self.args.time_horizon, self.args.num_envs),
            dtype=torch.bool,
            device=self.args.device,
        )
        values_t = torch.zeros(
            (self.args.time_horizon, self.args.num_envs), device=self.args.device
        )
        log_probs_t = torch.zeros(
            (self.args.time_horizon, self.args.num_envs), device=self.args.device
        )

        # collect observations
        next_obs, info = self.env.reset()
        next_obs = torch.from_numpy(next_obs).to(self.args.device)
        next_dones = torch.zeros(self.args.num_envs, device=self.args.device)
        # s0, a0, d0 -> s1, r0, d1
        for t in range(self.args.time_horizon):
            self.global_step += self.args.num_envs
            observations_t[t] = next_obs.unsqueeze(0)
            dones_t[t] = next_dones.unsqueeze(0)

            with torch.no_grad():
                actions, log_probs, _, values = self.agent.get_action_and_value(
                    next_obs
                )

            actions_t[t] = actions.unsqueeze(0)
            values_t[t] = values.unsqueeze(0)
            log_probs_t[t] = log_probs.unsqueeze(0)

            actions = actions.squeeze(0).cpu().numpy()
            next_obs, rewards, truncated, terminated, info = self.env.step([actions])
            next_obs, rewards, terminated, truncated = (
                torch.from_numpy(next_obs).to(self.args.device),
                torch.from_numpy(np.array(rewards)).to(self.args.device),
                torch.from_numpy(np.array(terminated)).to(self.args.device),
                torch.from_numpy(np.array(truncated)).to(self.args.device),
            )

            next_dones = torch.logical_or(terminated, truncated).float()

            rewards_t[t] = rewards.unsqueeze(0)

            if "episode" in info.keys():
                print(
                    f"Step: {self.global_step}, Episodic_return: {info['episode']['r']}"
                )
                self.logger.add_scalar(
                    "Stats/episodic_return", info["episode"]["r"], self.global_step
                )
                self.logger.add_scalar(
                    "Stats/episodic_length", info["episode"]["l"], self.global_step
                )

        # fill rollout buffer to sample from
        self.rollout_buffer.extend(
            observations_t,
            actions_t,
            rewards_t,
            dones_t,
            values_t,
            log_probs_t,
        )
        future_values = self.agent.get_value(next_obs)
        self.rollout_buffer.compute_rewards_and_advantages(future_values, next_dones)
        return

    def update(self):
        clip_fracs = []
        for step in range(self.args.num_epochs):
            for batch in self.rollout_buffer.batches(
                self.args.batch_size, shuffle=True
            ):
                observations = batch.observations.squeeze(1)
                actions = batch.actions.squeeze(1)
                advantages = batch.advantages.squeeze(1)
                returns = batch.returns.squeeze(1)
                old_values = batch.values.squeeze(1)
                old_logprobs = batch.log_probs.squeeze(1)

                if self.args.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                _, new_logprob, entropy, values_pred = self.agent.get_action_and_value(
                    observations, actions
                )

                # policy loss
                log_ratio = new_logprob - old_logprobs
                ratio = log_ratio.exp()
                policy_loss1 = -advantages * ratio
                policy_loss2 = -advantages * torch.clamp(
                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > self.args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                # critic training
                values_pred = values_pred.flatten()
                if self.args.clip_value_loss:
                    v_loss_unclipped = torch.square(values_pred - returns)
                    v_clipped = old_values + torch.clamp(
                        values_pred - old_values,
                        -self.args.clip_coef,
                        self.args.clip_coef,
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
                    + self.args.critic_loss_coef * critic_loss
                    - self.args.entropy_loss_coef * entropy_loss
                )

                self.optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.args.max_grad_norm
                )
                self.optim.step()

        y_pred, y_true = (
            self.rollout_buffer.values.cpu().numpy(),
            self.rollout_buffer.returns.cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.logger.add_scalar(
            "Stats/learning_rate", self.optim.param_groups[0]["lr"], self.global_step
        )
        self.logger.add_scalar("Loss/approx_kl", approx_kl.item(), self.global_step)
        self.logger.add_scalar(
            "Loss/old_approx_kl", old_approx_kl.item(), self.global_step
        )
        self.logger.add_scalar("Loss/clipfrac", np.mean(clip_fracs), self.global_step)
        self.logger.add_scalar("Loss/value_loss", critic_loss.item(), self.global_step)
        self.logger.add_scalar("Loss/actor_loss", policy_loss.item(), self.global_step)
        self.logger.add_scalar("Loss/entropy", entropy_loss.item(), self.global_step)
        self.logger.add_scalar(
            "Loss/explained_variance", explained_var, self.global_step
        )


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
        log_prob = log_prob.sum(1)
        entropy = probs.entropy().sum(1)
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
    # for k, v in vars(args).items():
    #     print(k, ": ", v)

    env = make_env(config.env_id)
    env = gym.vector.SyncVectorEnv([lambda: env])

    model = PPO(env, config)
    model.train()

    env.close()
    # model_path = Path(f"models/{id}/ppo.pth")
    # model_path.mkdir(parents=True, exist_ok=True)
    # torch.save(model.agent.state_dict(), model_path)
