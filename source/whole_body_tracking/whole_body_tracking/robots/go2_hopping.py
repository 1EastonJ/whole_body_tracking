import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from whole_body_tracking.robots.go2 import (
    GO2_CALF_JOINT_NAMES,
    GO2_CSV_JOINT_NAMES,
    GO2_FOOT_BODY_NAMES,
    GO2_HIP_AND_THIGH_JOINT_NAMES,
    GO2_NON_FOOT_CONTACT_BODY_NAMES,
    GO2_TRACKING_ANCHOR_BODY_NAME,
)


GO2_HOPPING_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

GO2_HOPPING_ACTION_SCALE = 0.25
GO2_HOPPING_TERMINATION_BODY_NAMES = (GO2_TRACKING_ANCHOR_BODY_NAME,)
GO2_HOPPING_PENALIZED_CONTACT_BODY_NAMES = tuple(
    name for name in GO2_NON_FOOT_CONTACT_BODY_NAMES if "thigh" in name or "calf" in name
)


GO2_HOPPING_CFG = UNITREE_GO2_CFG.replace(
    spawn=UNITREE_GO2_CFG.spawn.replace(
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos=GO2_HOPPING_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "go2_hip_and_thigh": DCMotorCfg(
            joint_names_expr=list(GO2_HIP_AND_THIGH_JOINT_NAMES),
            stiffness=20.0,
            damping=0.5,
            armature=0.0,
            effort_limit=23.7,
            effort_limit_sim=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            velocity_limit_sim=30.1,
            friction=0.0,
        ),
        "go2_calf": DCMotorCfg(
            joint_names_expr=list(GO2_CALF_JOINT_NAMES),
            stiffness=20.0,
            damping=0.5,
            armature=0.0,
            effort_limit=45.43,
            effort_limit_sim=45.43,
            saturation_effort=45.43,
            velocity_limit=15.70,
            velocity_limit_sim=15.70,
            friction=0.0,
        ),
    },
)

GO2_HOPPING_TRAMPOLINE_CFG = GO2_HOPPING_CFG.replace(
    spawn=GO2_HOPPING_CFG.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
)

GO2_HOPPING_ACTION_SCALE_MAP = {name: GO2_HOPPING_ACTION_SCALE for name in GO2_CSV_JOINT_NAMES}
GO2_HOPPING_FOOT_BODY_NAMES = GO2_FOOT_BODY_NAMES
