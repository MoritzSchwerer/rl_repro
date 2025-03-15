import torch
from typing import Tuple, Iterator, Dict

from collections import namedtuple


def assert_not_nan(value: torch.Tensor):
    if value.isnan().any():
        print(value)
        assert False, "Found nans"


def log_model_gradients(logger, model, name, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(f"{name}Grad/" + tag, value.grad.cpu(), step)


Batch = namedtuple(
    "Batch",
    [
        "observations",
        "actions",
        "rewards",
        "returns",
        "dones",
        "values",
        "log_probs",
        "advantages",
    ],
)


class ReplayBuffer:
    observations: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    new_observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    probs: torch.Tensor

    def __init__(
        self,
        size: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        acts_shape: Tuple[int, ...],
        gamma: float = 0.99,
        lmda: float = 0.95,
        device="cpu",
    ):
        self.size = size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.acts_shape = acts_shape
        self.gamma = gamma
        self.lmda = lmda
        self.device = device

        self.observations = torch.empty(
            (self.size, self.num_envs) + self.obs_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.empty(
            (self.size, self.num_envs) + self.acts_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.empty((self.size, self.num_envs), device=self.device)
        self.returns = torch.empty((self.size, self.num_envs), device=self.device)
        self.advantages = torch.empty((self.size, self.num_envs), device=self.device)
        self.dones = torch.empty((self.size, self.num_envs), device=self.device)
        self.values = torch.empty((self.size, self.num_envs), device=self.device)
        self.log_probs = torch.empty((self.size, self.num_envs), device=self.device)
        self.index = 0
        self.ready_to_generate = False

    def clear(self) -> None:
        self.index = 0
        self.ready_to_generate = False

    def is_full(self) -> bool:
        return self.index == self.size

    def extend(self, observations, actions, rewards, dones, values, log_probs):
        batch_size = observations.shape[0]
        assert (
            observations.shape[0]
            == actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == values.shape[0]
            == log_probs.shape[0]
        )
        assert (
            observations.shape[1]
            == actions.shape[1]
            == rewards.shape[1]
            == dones.shape[1]
            == values.shape[1]
            == log_probs.shape[1]
            == self.num_envs
        )
        start = self.index
        end = self.index + batch_size
        assert end <= self.size, "End index should be smaller than size"
        self.observations[start:end] = observations
        self.actions[start:end] = actions
        self.rewards[start:end] = rewards
        self.dones[start:end] = dones
        self.values[start:end] = values
        self.log_probs[start:end] = log_probs
        self.index += batch_size

    def shuffle(self) -> None:
        """Shuffles the valid elements of the buffer."""
        indices = torch.randperm(self.index)
        self.observations[: self.index] = self.observations[indices]
        self.actions[: self.index] = self.actions[indices]
        self.rewards[: self.index] = self.rewards[indices]
        self.returns[: self.index] = self.returns[indices]
        self.dones[: self.index] = self.dones[indices]
        self.values[: self.index] = self.values[indices]
        self.log_probs[: self.index] = self.log_probs[indices]
        self.advantages[: self.index] = self.advantages[indices]

    def batches(self, batch_size: int, shuffle: bool = True) -> Iterator[Batch]:
        """Yields batches of elements from the buffer."""
        assert (
            self.ready_to_generate
        ), "Need to call compute_rewards_and_advantages first"
        if shuffle:
            self.shuffle()

        for start in range(0, self.index, batch_size):
            end = min(start + batch_size, self.index)
            yield Batch(
                self.observations[start:end],
                self.actions[start:end],
                self.rewards[start:end],
                self.returns[start:end],
                self.dones[start:end],
                self.values[start:end],
                self.log_probs[start:end],
                self.advantages[start:end],
            )

    def compute_rewards_and_advantages(
        self, last_values: torch.Tensor, next_dones: torch.Tensor
    ):
        with torch.no_grad():
            gae = 0
            for t in reversed(range(self.size)):
                if t == self.size - 1:
                    next_non_terminal = 1.0 - next_dones
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_values = self.values[t + 1]

                # delta = r_t + gamma * v_{t+1} - v_t
                delta = (
                    self.rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - self.values[t]
                )
                gae = delta + self.gamma * self.lmda * next_non_terminal * gae
                self.advantages[t] = gae
            self.returns = self.advantages + self.values
            self.ready_to_generate = True
