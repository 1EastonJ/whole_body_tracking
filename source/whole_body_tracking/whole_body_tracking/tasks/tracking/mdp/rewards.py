from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, yaw_quat

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_global_anchor_yaw_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_yaw_quat = yaw_quat(command.anchor_quat_w)
    robot_anchor_yaw_quat = yaw_quat(command.robot_anchor_quat_w)
    error = quat_error_magnitude(anchor_yaw_quat, robot_anchor_yaw_quat) ** 2
    return torch.exp(-error / std**2)


def motion_global_anchor_yaw_penalty(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_yaw_quat = yaw_quat(command.anchor_quat_w)
    robot_anchor_yaw_quat = yaw_quat(command.robot_anchor_quat_w)
    error = quat_error_magnitude(anchor_yaw_quat, robot_anchor_yaw_quat) ** 2
    return 1.0 - torch.exp(-error / std**2)


def motion_global_anchor_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def anchor_height_below_reference_penalty(
    env: ManagerBasedRLEnv, command_name: str, margin: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    deficit = command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2] - margin
    return torch.clamp(deficit, min=0.0)



def joint_mirror_symmetry_l2(
    env: ManagerBasedRLEnv,
    left_joint_names: list[str],
    right_joint_names: list[str],
    mirror_signs: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_weight: float = 0.0,
) -> torch.Tensor:
    """Penalize left/right joint asymmetry for sagittal-plane motions.

    ``mirror_signs`` is ``-1`` for mirrored hip ab/adduction joints and ``1`` for
    joints that should match directly, such as thigh and calf flexion.
    """
    asset = env.scene[asset_cfg.name]
    joint_name_to_id = {name: idx for idx, name in enumerate(asset.data.joint_names)}
    left_ids = torch.tensor([joint_name_to_id[name] for name in left_joint_names], device=asset.data.joint_pos.device)
    right_ids = torch.tensor([joint_name_to_id[name] for name in right_joint_names], device=asset.data.joint_pos.device)
    signs = torch.tensor(mirror_signs, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)

    pos_error = asset.data.joint_pos[:, left_ids] - signs * asset.data.joint_pos[:, right_ids]
    penalty = torch.mean(torch.square(pos_error), dim=1)
    if vel_weight > 0.0:
        vel_error = asset.data.joint_vel[:, left_ids] - signs * asset.data.joint_vel[:, right_ids]
        penalty = penalty + vel_weight * torch.mean(torch.square(vel_error), dim=1)
    return penalty


def joint_near_lower_limit_penalty(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    lower_limit_margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints that approach their lower position limits.

    The penalty is zero while a joint stays farther than ``lower_limit_margin``
    away from its lower limit, and increases smoothly once it enters that band.
    """
    asset = env.scene[asset_cfg.name]
    joint_name_to_id = {name: idx for idx, name in enumerate(asset.data.joint_names)}
    joint_ids = torch.tensor([joint_name_to_id[name] for name in joint_names], device=asset.data.joint_pos.device)
    lower_limits = asset.data.joint_pos_limits[:, joint_ids, 0]
    dist_to_lower = asset.data.joint_pos[:, joint_ids] - lower_limits
    normalized_violation = torch.clamp((lower_limit_margin - dist_to_lower) / lower_limit_margin, min=0.0)
    return torch.mean(torch.square(normalized_violation), dim=1)
