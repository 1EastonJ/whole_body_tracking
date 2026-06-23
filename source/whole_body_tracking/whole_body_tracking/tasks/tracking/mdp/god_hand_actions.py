from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


class GodHandWrenchAction(ActionTerm):
    """Apply a 6-DoF external wrench action to one articulation body."""

    cfg: "GodHandWrenchActionCfg"

    def __init__(self, cfg: "GodHandWrenchActionCfg", env):
        super().__init__(cfg, env)

        self._body_ids, self._body_names = self._asset.find_bodies([cfg.body_name], preserve_order=True)
        if len(self._body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching {cfg.body_name!r} for god hand action, "
                f"but found {self._body_names}."
            )

        self._raw_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torques = torch.zeros_like(self._forces)
        self._force_scale = torch.tensor(cfg.force_scale, device=self.device).view(1, 3)
        self._torque_scale = torch.tensor(cfg.torque_scale, device=self.device).view(1, 3)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self.device)
        self._processed_actions[:, :3] = self._raw_actions[:, :3] * self._force_scale
        self._processed_actions[:, 3:] = self._raw_actions[:, 3:] * self._torque_scale

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._forces[env_ids] = 0.0
        self._torques[env_ids] = 0.0

    def apply_actions(self):
        self._forces[:, 0, :] = self._processed_actions[:, :3]
        self._torques[:, 0, :] = self._processed_actions[:, 3:]
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            body_ids=self._body_ids,
            is_global=self.cfg.is_global,
        )


@configclass
class GodHandWrenchActionCfg(ActionTermCfg):
    """Configuration for a 6-DoF god-hand wrench action.

    The action order is ``[Fx, Fy, Fz, tx, ty, tz]``.
    """

    class_type: type = GodHandWrenchAction
    body_name: str = "pelvis"
    force_scale: tuple[float, float, float] = (300.0, 300.0, 300.0)
    torque_scale: tuple[float, float, float] = (50.0, 50.0, 50.0)
    is_global: bool = False
