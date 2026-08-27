# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp

import pydantic
import torch
from torch.amp import autocast

from ..base_model import load_model
from ..fb.model import FBModel, FBModelArchiConfig, FBModelConfig
from ..nn_models import DiscriminatorArchiConfig, ForwardArchiConfig, RewardNormalizerConfig


class FBcprAuxModelArchiConfig(FBModelArchiConfig):
    critic: ForwardArchiConfig = pydantic.Field(ForwardArchiConfig(), discriminator="name")
    discriminator: DiscriminatorArchiConfig = pydantic.Field(DiscriminatorArchiConfig(), discriminator="name")
    aux_critic: ForwardArchiConfig = ForwardArchiConfig()


class FBcprAuxModelConfig(FBModelConfig):
    name: tp.Literal["FBcprAuxModel"] = "FBcprAuxModel"
    archi: FBcprAuxModelArchiConfig = FBcprAuxModelArchiConfig()
    norm_aux_reward: RewardNormalizerConfig = RewardNormalizerConfig()

    @property
    def object_class(self):
        return FBcprAuxModel


def config_mutator_for_wrong_aux_critic(loaded_config: dict) -> dict:
    """Old versions of FBcprAuxModel had a bug where aux_critic was using config of critic instead of aux_critic"""
    loaded_config["archi"]["aux_critic"] = copy.deepcopy(loaded_config["archi"]["critic"])
    return loaded_config


class FBcprAuxModel(FBModel):
    config_class = FBcprAuxModelConfig

    def __init__(self, obs_space, action_dim: int, cfg: FBcprAuxModelConfig):
        # NOTE for future: if we inherit models, we need to make sure that the cfg we pass in (which is wrong)
        #      can still be used to build the underlying models
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: FBcprAuxModelConfig = cfg
        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)
        self._critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._aux_critic = cfg.archi.aux_critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._aux_reward_normalizer = cfg.norm_aux_reward.build()

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_critic = copy.deepcopy(self._critic)
        self._target_aux_critic = copy.deepcopy(self._aux_critic)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._discriminator(self._normalize(obs), z)

    @torch.no_grad()
    def aux_critic(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._aux_critic(self._normalize(obs), z, action)

    @classmethod
    def load(cls, path: str, device: str | None = None, strict: bool = True):
        try:
            model = load_model(path, device, strict=strict, config_class=cls.config_class)
        except Exception:
            # try mutating the config to fix old bug
            model = load_model(
                path,
                device,
                strict=strict,
                config_class=cls.config_class,
                config_mutator=config_mutator_for_wrong_aux_critic,
            )
        return model
