from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from whole_body_tracking.utils.trampoline_deformable import TRAMPOLINE_PIN_RADIUS, build_trampoline_kinematic_targets


class TrampolinePinningAction(ActionTerm):
    """Continuously rewrite deformable trampoline kinematic targets."""

    cfg: "TrampolinePinningActionCfg"

    def __init__(self, cfg: "TrampolinePinningActionCfg", env):
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._pin_radius = cfg.pin_radius
        self._targets = torch.zeros_like(self._asset.data.nodal_kinematic_target)
        self._pinned_mask = torch.zeros_like(self._asset.data.nodal_kinematic_target[..., 3], dtype=torch.bool)
        self._center_node_ids = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        self.refresh_targets()

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.to(self.device)
        self._processed_actions = self._raw_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids=env_ids)
        self.refresh_targets(env_ids)
        self.write_targets(env_ids)

    def apply_actions(self):
        self.write_targets()

    def refresh_targets(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_tensor = self._resolve_env_ids(env_ids)
        index = slice(None) if env_ids_tensor is None else env_ids_tensor

        targets, pinned_mask, center_node_ids = build_trampoline_kinematic_targets(
            self._asset.data.default_nodal_state_w[index],
            self._asset.data.nodal_kinematic_target[index],
            pin_radius=self._pin_radius,
        )
        self._targets[index] = targets
        self._pinned_mask[index] = pinned_mask
        self._center_node_ids[index] = center_node_ids

    def write_targets(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_tensor = self._resolve_env_ids(env_ids)
        if env_ids_tensor is None:
            self._asset.write_nodal_kinematic_target_to_sim(self._targets)
        else:
            self._asset.write_nodal_kinematic_target_to_sim(self._targets[env_ids_tensor], env_ids=env_ids_tensor)

    def _resolve_env_ids(self, env_ids: Sequence[int] | None = None) -> torch.Tensor | None:
        if env_ids is None:
            return None
        if isinstance(env_ids, slice):
            if env_ids != slice(None):
                raise ValueError(f"Unsupported slice for env_ids: {env_ids}")
            return None
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class TrampolinePinningActionCfg(ActionTermCfg):
    """Configuration for continuous deformable trampoline rim pinning."""

    class_type: type = TrampolinePinningAction
    pin_radius: float = TRAMPOLINE_PIN_RADIUS
