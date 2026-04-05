from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.go2 import (
    GO2_ACTION_SCALE,
    GO2_CFG,
    GO2_CSV_JOINT_NAMES,
    GO2_FOOT_BODY_NAMES,
    GO2_NON_FOOT_CONTACT_BODY_NAMES,
    GO2_TRACKING_ANCHOR_BODY_NAME,
)
from whole_body_tracking.robots.go2_hopping import (
    GO2_HOPPING_ACTION_SCALE_MAP,
    GO2_HOPPING_CFG,
    GO2_HOPPING_TRAMPOLINE_CFG,
)
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_PIN_WIDTH,
    TRAMPOLINE_RADIUS,
    TRAMPOLINE_THICKNESS,
    TRAMPOLINE_TOP_Z,
    make_trampoline_cfg,
)


def _ordered_action_scale(action_scale_map: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(action_scale_map[joint_name]) for joint_name in GO2_CSV_JOINT_NAMES)


GO2_HOPPING_PENALIZED_CONTACT_BODY_NAMES = tuple(
    name for name in GO2_NON_FOOT_CONTACT_BODY_NAMES if "thigh" in name or "calf" in name
)


@configclass
class CommandRangesCfg:
    lin_vel_x = (-1.0, 1.0)
    lin_vel_y = (-1.0, 1.0)
    ang_vel_yaw = (-1.0, 1.0)
    heading = (-3.14, 3.14)


@configclass
class CommandsCfg:
    curriculum = True
    max_curriculum = 2.0
    num_commands = 4
    resampling_time = 5.0
    heading_command = False
    command_xy_deadzone = 0.1
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class DomainRandCfg:
    randomize_friction = True
    friction_range = (0.4, 0.8)

    push_robots = True
    push_interval_s = 4.0
    max_push_vel_xy = 0.4
    max_push_ang_vel = 0.6

    randomize_base_mass = True
    added_base_mass_range = (-1.0, 1.0)

    randomize_link_mass = True
    multiplied_link_mass_range = (0.9, 1.1)

    randomize_base_com = True
    added_base_com_range = (-0.02, 0.02)

    randomize_pd_gains = True
    stiffness_multiplier_range = (0.9, 1.1)
    damping_multiplier_range = (0.9, 1.1)

    randomize_motor_zero_offset = True
    motor_zero_offset_range = (-0.035, 0.035)

    add_obs_latency = True
    randomize_obs_motor_latency = True
    randomize_obs_imu_latency = True
    range_obs_motor_latency = (1, 3)
    range_obs_imu_latency = (1, 3)

    add_cmd_action_latency = True
    randomize_cmd_action_latency = True
    range_cmd_action_latency = (1, 3)


@configclass
class RewardScalesCfg:
    termination = -0.0
    tracking_lin_vel = 2.0
    tracking_ang_vel = 2.0
    lin_vel_z = 0.05
    ang_vel_xy = 0.2
    orientation = 0.6
    torques = -0.0002
    dof_vel = -0.0
    dof_acc = -5.5e-4
    base_height = 1.0
    feet_air_time = 1.0
    collision = -1.0
    feet_stumble = -0.0
    action_rate = -0.01
    default_pos = -0.1
    default_hip_pos = 0.3
    feet_contact_forces = -0.01
    jump = 2.0
    feet_clearance = 0.5


@configclass
class RewardsCfg:
    scales: RewardScalesCfg = RewardScalesCfg()
    only_positive_rewards = False
    tracking_sigma = 0.25
    soft_dof_pos_limit = 0.9
    soft_dof_vel_limit = 1.0
    soft_torque_limit = 1.0
    base_height_target = 0.3
    max_contact_force = 100.0
    cycle_time = 1.5
    target_feet_height = 0.05


@configclass
class ObsScalesCfg:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    height_measurements = 5.0
    quat = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations = 100.0
    clip_actions = 100.0


@configclass
class NoiseScalesCfg:
    dof_pos = 0.01
    dof_vel = 1.5
    lin_vel = 0.1
    ang_vel = 0.2
    gravity = 0.05
    quat = 0.1
    height_measurements = 0.1


@configclass
class NoiseCfg:
    add_noise = True
    noise_level = 1.0
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class Go2HoppingFlatEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(10.0, 0.0, 6.0),
        lookat=(11.0, 5.0, 3.0),
        origin_type="world",
    )
    episode_length_s = 24.0
    decimation = 4
    action_space = 12
    observation_space = 470
    state_space = 210

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
        ),
    )
    terrain: TerrainImporterCfg | None = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    trampoline: DeformableObjectCfg | None = None
    trampoline_radius = TRAMPOLINE_RADIUS
    trampoline_pin_width = TRAMPOLINE_PIN_WIDTH
    trampoline_thickness = TRAMPOLINE_THICKNESS
    trampoline_surface_height = TRAMPOLINE_TOP_Z
    trampoline_robot_clearance = 0.0
    usable_radius = TRAMPOLINE_RADIUS
    use_plain_trampoline_visual = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    robot = GO2_HOPPING_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    action_scale = _ordered_action_scale(GO2_HOPPING_ACTION_SCALE_MAP)
    foot_body_names = GO2_FOOT_BODY_NAMES
    penalized_contact_body_names = GO2_HOPPING_PENALIZED_CONTACT_BODY_NAMES
    termination_body_names = (GO2_TRACKING_ANCHOR_BODY_NAME,)
    non_foot_contact_body_names = GO2_NON_FOOT_CONTACT_BODY_NAMES
    anchor_body_name = GO2_TRACKING_ANCHOR_BODY_NAME
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )

    commands: CommandsCfg = CommandsCfg()
    domain_rand: DomainRandCfg = DomainRandCfg()
    rewards: RewardsCfg = RewardsCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg()


def _is_play_mode() -> bool:
    return os.environ.get("WHOLE_BODY_TRACKING_PLAY_MODE") == "1"


def _apply_play_overrides(cfg: Go2HoppingFlatEnvCfg) -> Go2HoppingFlatEnvCfg:
    cfg.episode_length_s = float(1.0e9)
    cfg.noise.add_noise = False
    cfg.domain_rand.push_robots = False
    cfg.viewer.eye = (1.8, 1.8, 1.2)
    cfg.viewer.lookat = (0.0, 0.0, 0.3)
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.body_name = cfg.anchor_body_name
    return cfg


def go2_hopping_flat_env_cfg() -> Go2HoppingFlatEnvCfg:
    cfg = Go2HoppingFlatEnvCfg()
    if _is_play_mode():
        cfg = _apply_play_overrides(cfg)
    return cfg


def go2_hopping_trampoline_env_cfg() -> Go2HoppingFlatEnvCfg:
    cfg = Go2HoppingFlatEnvCfg()
    cfg.terrain = None
    cfg.sim.physx.max_position_iteration_count = 8
    cfg.sim.physx.max_velocity_iteration_count = 4
    cfg.scene.replicate_physics = False
    cfg.scene.env_spacing = max(float(cfg.scene.env_spacing), 2.0 * float(cfg.trampoline_radius) + 2.0)
    cfg.robot = GO2_HOPPING_TRAMPOLINE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cfg.trampoline = make_trampoline_cfg(
        "/World/envs/env_.*/Trampoline",
        center_z=float(cfg.trampoline_surface_height) - 0.5 * float(cfg.trampoline_thickness),
        debug_vis=False,
    )
    cfg.usable_radius = float(cfg.trampoline_radius)
    cfg.use_plain_trampoline_visual = False
    if _is_play_mode():
        cfg = _apply_play_overrides(cfg)
    return cfg


def go2_hopping_trampoline_tracking_cfg_env_cfg() -> Go2HoppingFlatEnvCfg:
    cfg = go2_hopping_trampoline_env_cfg()
    cfg.robot = GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cfg.action_scale = _ordered_action_scale(GO2_ACTION_SCALE)

    cfg.commands.curriculum = True
    cfg.commands.resampling_time = 6.0
    cfg.commands.command_xy_deadzone = 0.0
    cfg.commands.ranges.lin_vel_x = (0.4, 1.0)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_yaw = (0.0, 0.0)
    cfg.commands.ranges.heading = (0.0, 0.0)

    cfg.rewards.scales.tracking_lin_vel = 1.5
    cfg.rewards.scales.tracking_ang_vel = 0.0
    cfg.rewards.scales.orientation = 1.0
    cfg.rewards.scales.dof_acc = -1.0e-4
    cfg.rewards.scales.action_rate = -0.002
    cfg.rewards.scales.default_pos = -0.02
    cfg.rewards.scales.base_height = 2.0
    cfg.rewards.scales.feet_air_time = 5.0
    cfg.rewards.scales.jump = 8.0
    cfg.rewards.scales.feet_clearance = 3.0
    cfg.rewards.scales.collision = -3.0
    cfg.rewards.base_height_target = 0.35
    cfg.rewards.cycle_time = 1.0
    cfg.rewards.target_feet_height = 0.12

    cfg.usable_radius = min(float(cfg.trampoline_radius), 4.0)
    return cfg


def go2_hopping_flat_tracking_cfg_env_cfg() -> Go2HoppingFlatEnvCfg:
    cfg = Go2HoppingFlatEnvCfg()
    cfg.robot = GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cfg.action_scale = _ordered_action_scale(GO2_ACTION_SCALE)
    if _is_play_mode():
        cfg = _apply_play_overrides(cfg)
    return cfg
