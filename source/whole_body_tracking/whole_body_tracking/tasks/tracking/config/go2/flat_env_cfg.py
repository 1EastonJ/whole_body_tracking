from __future__ import annotations

import os

from isaaclab.assets import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.go2 import (
    GO2_ACTION_SCALE,
    GO2_CFG,
    GO2_FOOT_BODY_NAMES,
    GO2_NON_FOOT_CONTACT_BODY_NAMES,
    GO2_TRACKING_ANCHOR_BODY_NAME,
    GO2_TRACKING_BODY_NAMES,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MySceneCfg, TrackingEnvCfg
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_PIN_WIDTH,
    TRAMPOLINE_RADIUS,
    TRAMPOLINE_THICKNESS,
    TRAMPOLINE_TOP_Z,
    make_trampoline_cfg,
)


def _is_play_mode() -> bool:
    return os.environ.get('WHOLE_BODY_TRACKING_PLAY_MODE') == '1'


def _apply_play_overrides(cfg: TrackingEnvCfg) -> TrackingEnvCfg:
    cfg.episode_length_s = int(1e9)
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.sampling_mode = 'start'
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    # Keep the play camera static so manual mouse control is not overridden by asset tracking.
    cfg.viewer.origin_type = 'world'
    return cfg


@configclass
class Go2TrampolineSceneCfg(MySceneCfg):
    trampoline: DeformableObjectCfg = make_trampoline_cfg(
        "{ENV_REGEX_NS}/Trampoline",
        center_z=float(TRAMPOLINE_TOP_Z) - 0.5 * float(TRAMPOLINE_THICKNESS),
        debug_vis=False,
    )




@configclass
class Go2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = GO2_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.viewer.body_name = GO2_TRACKING_ANCHOR_BODY_NAME

        self.actions.joint_pos.scale = GO2_ACTION_SCALE

        self.commands.motion.anchor_body_name = GO2_TRACKING_ANCHOR_BODY_NAME
        self.commands.motion.body_names = list(GO2_TRACKING_BODY_NAMES)

        self.events.base_com.params['asset_cfg'] = SceneEntityCfg('robot', body_names=GO2_TRACKING_ANCHOR_BODY_NAME)
        self.events.physics_material.func = mdp.randomize_rigid_body_material_shared
        self.events.physics_material.params['asset_cfg'] = SceneEntityCfg('robot', body_names=list(GO2_FOOT_BODY_NAMES))
        self.events.physics_material.params['static_friction_range'] = (0.3, 1.2)
        self.events.physics_material.params['dynamic_friction_range'] = (0.3, 1.2)
        self.events.physics_material.params['restitution_range'] = (0.0, 0.0)
        self.events.physics_material.params['make_consistent'] = True

        self.rewards.motion_global_anchor_ori.weight = 1.5
        self.rewards.motion_body_ang_vel.weight = 2.0
        self.rewards.motion_body_ang_vel.params['std'] = 6.28
        # mjlab's Go2 setup keeps the self-collision cost effectively inactive for this task.
        # Penalizing all non-foot PhysX contacts is harsher and tends to make aerial phases too conservative.
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.undesired_contacts.params['sensor_cfg'] = SceneEntityCfg(
            'contact_forces', body_names=list(GO2_NON_FOOT_CONTACT_BODY_NAMES)
        )

        self.terminations.anchor_pos.params['threshold'] = 0.5
        self.terminations.ee_body_pos.params['threshold'] = 0.6
        self.terminations.ee_body_pos.params['body_names'] = list(GO2_FOOT_BODY_NAMES)


@configclass
class Go2FlatNoStateEstimationEnvCfg(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Go2TrampolineNoStateEstimationEnvCfg(Go2FlatNoStateEstimationEnvCfg):
    scene: Go2TrampolineSceneCfg = Go2TrampolineSceneCfg(
        num_envs=4096,
        env_spacing=max(2.5, 2.0 * float(TRAMPOLINE_RADIUS) + 2.0),
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = None
        self.scene.replicate_physics = False
        self.scene.env_spacing = max(float(self.scene.env_spacing), 2.0 * float(TRAMPOLINE_RADIUS) + 2.0)
        self.events.reset_trampoline = EventTerm(
            func=mdp.reset_deformable_trampoline_event,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("trampoline"),
                "pin_width": float(TRAMPOLINE_PIN_WIDTH),
            },
        )
        self.terminations.out_of_trampoline = DoneTerm(
            func=mdp.root_xy_too_far_from_origin,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "threshold": max(0.0, float(TRAMPOLINE_RADIUS) - 0.25),
            },
        )
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_out_of_bounds,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "min_height": float(TRAMPOLINE_TOP_Z) - 1.0,
                "max_height": float(TRAMPOLINE_TOP_Z) + 1.25,
            },
        )


def go2_trampoline_no_state_estimation_env_cfg() -> Go2TrampolineNoStateEstimationEnvCfg:
    cfg = Go2TrampolineNoStateEstimationEnvCfg()
    if _is_play_mode():
        cfg = _apply_play_overrides(cfg)
    return cfg


def go2_flat_env_cfg() -> Go2FlatEnvCfg:
    cfg = Go2FlatEnvCfg()
    if _is_play_mode():
        cfg = _apply_play_overrides(cfg)
    return cfg


def go2_flat_no_state_estimation_env_cfg() -> Go2FlatNoStateEstimationEnvCfg:
    cfg = Go2FlatNoStateEstimationEnvCfg()
    if _is_play_mode():
        cfg = _apply_play_overrides(cfg)
    return cfg


# Backward-compatible alias while the public task name uses "No-State-Estimation".
Go2FlatWoStateEstimationEnvCfg = Go2FlatNoStateEstimationEnvCfg


def go2_flat_wo_state_estimation_env_cfg() -> Go2FlatNoStateEstimationEnvCfg:
    return go2_flat_no_state_estimation_env_cfg()
