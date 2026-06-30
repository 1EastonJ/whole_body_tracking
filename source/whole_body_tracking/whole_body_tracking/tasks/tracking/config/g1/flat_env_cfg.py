from isaaclab.assets import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ActionsCfg, MySceneCfg, TrackingEnvCfg
from whole_body_tracking.utils.trampoline_deformable import (
    MIXED_RESET_DROP_HEIGHT_RANGE,
    MIXED_RESET_STATIC_HEIGHT_OFFSET,
    TRAMPOLINE_DR_DAMPING_SCALE_RANGE,
    TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE,
    TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE,
    TRAMPOLINE_DR_MASS_RANGE,
    TRAMPOLINE_DR_POISSONS_RATIO_RANGE,
    TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    TRAMPOLINE_PIN_RADIUS,
    TRAMPOLINE_RADIUS,
    TRAMPOLINE_THICKNESS,
    TRAMPOLINE_TOP_Z,
    make_trampoline_cfg,
)


@configclass
class G1TrampolineSceneCfg(MySceneCfg):
    trampoline: DeformableObjectCfg = make_trampoline_cfg(
        "{ENV_REGEX_NS}/Trampoline",
        center_z=float(TRAMPOLINE_TOP_Z) - 0.5 * float(TRAMPOLINE_THICKNESS),
        debug_vis=False,
    )


@configclass
class G1TrampolineActionsCfg(ActionsCfg):
    trampoline_pin = mdp.TrampolinePinningActionCfg(
        asset_name="trampoline",
        pin_radius=float(TRAMPOLINE_PIN_RADIUS),
    )


@configclass
class G1GodHandActionsCfg(ActionsCfg):
    god_hand = mdp.GodHandWrenchActionCfg(
        asset_name="robot",
        body_name="pelvis",
        force_scale=(300.0, 300.0, 300.0),
        torque_scale=(50.0, 50.0, 50.0),
        is_global=False,
    )


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


@configclass
class G1TrampolineEnvCfg(G1FlatEnvCfg):
    scene: G1TrampolineSceneCfg = G1TrampolineSceneCfg(
        num_envs=4096,
        env_spacing=max(2.5, 2.0 * float(TRAMPOLINE_RADIUS) + 2.0),
        replicate_physics=False,
    )
    actions: G1TrampolineActionsCfg = G1TrampolineActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = None
        self.scene.replicate_physics = False
        self.scene.env_spacing = max(float(self.scene.env_spacing), 2.0 * float(TRAMPOLINE_RADIUS) + 2.0)
        self.commands.motion.pose_range = dict(self.commands.motion.pose_range)
        self.commands.motion.pose_range["z"] = (
            float(MIXED_RESET_STATIC_HEIGHT_OFFSET) + float(MIXED_RESET_DROP_HEIGHT_RANGE[0]),
            float(MIXED_RESET_STATIC_HEIGHT_OFFSET) + float(MIXED_RESET_DROP_HEIGHT_RANGE[1]),
        )
        self.events.reset_trampoline = EventTerm(
            func=mdp.reset_deformable_trampoline_event,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("trampoline"),
                "pin_radius": float(TRAMPOLINE_PIN_RADIUS),
                "randomize_material": False,
                "youngs_modulus_range": TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
                "mass_range": TRAMPOLINE_DR_MASS_RANGE,
                "dynamic_friction_range": TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE,
                "elasticity_damping_range": TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE,
                "damping_scale_range": TRAMPOLINE_DR_DAMPING_SCALE_RANGE,
                "poissons_ratio_range": TRAMPOLINE_DR_POISSONS_RATIO_RANGE,
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
                "max_height": float(TRAMPOLINE_TOP_Z) + 2.5,
            },
        )


@configclass
class G1FlatGodHandEnvCfg(G1FlatEnvCfg):
    actions: G1GodHandActionsCfg = G1GodHandActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = (
            "/inspire/hdd/project/leverage-robot/ky26214/whole_body_tracking/datag1/npz/video_005_v2r.npz"
        )


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
