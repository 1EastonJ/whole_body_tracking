"""Unified trampoline smoke script for G1, Go2, or ball actors.

This script merges the common use-cases from:

- ``trampoline.py``: ball + built-in deformable trampoline
- ``trampoline_spring.py``: ball + custom spring contact model
- ``load_g1_trampoline.py``: G1 + built-in deformable trampoline
- ``load_g1.py``: G1 reset / default-pose holding logic

The two main switches are:

- ``--actor {g1,go2,ball}``
- ``--trampoline_mode {builtin,spring}``

Notes:

- ``builtin`` uses Isaac Lab's deformable body support with pinned rim nodes.
- ``spring`` uses a custom flat support model:
  - ``ball``: a vertical spring-damper force on the rigid ball
  - ``g1``: the existing point-foot custom contact model on the two ankle links
  - ``go2``: a point-foot custom contact model on the four foot links
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher
import pathlib
import sys

import numpy as np


def _add_repo_source_to_path():
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / "source" / "whole_body_tracking"
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_add_repo_source_to_path()

# This script only needs robot/assets utilities. Skipping task auto-registration
# avoids unrelated Gym config import failures during smoke tests.
os.environ.setdefault("WHOLE_BODY_TRACKING_IMPORT_TASKS", "0")


parser = argparse.ArgumentParser(description="Load G1, Go2, or ball on a built-in or custom trampoline.")
parser.add_argument("--actor", type=str, choices=("g1", "go2", "ball"), default="g1", help="Actor placed on the trampoline.")
parser.add_argument(
    "--trampoline_mode",
    type=str,
    choices=("builtin", "spring"),
    default="builtin",
    help="Trampoline implementation to use.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene instances to spawn.")
parser.add_argument("--env_spacing", type=float, default=None, help="Environment spacing. Uses a mode-dependent default.")
parser.add_argument(
    "--passive",
    action="store_true",
    default=False,
    help="If set for robot actors, do not command the default standing pose.",
)
parser.add_argument(
    "--robot_reset_mode",
    type=str,
    choices=("stand", "drop"),
    default="stand",
    help="Robot reset style: place the robot in its default standing pose or drop it from above the trampoline.",
)
parser.add_argument(
    "--drop_height",
    type=float,
    default=2.5,
    help="Additional root height above the normal robot reset pose when using --robot_reset_mode drop.",
)
parser.add_argument("--drop_vx", type=float, default=0.0, help="Initial robot x velocity for drop reset.")
parser.add_argument("--drop_vy", type=float, default=0.0, help="Initial robot y velocity for drop reset.")
parser.add_argument("--drop_vz", type=float, default=0.0, help="Initial robot z velocity for drop reset.")
parser.add_argument(
    "--reset_interval",
    type=int,
    default=None,
    help="Number of simulation steps between resets. Uses an actor-dependent default when omitted.",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=120,
    help="Number of simulation steps between status prints.",
)
parser.add_argument(
    "--surface_height",
    type=float,
    default=None,
    help="Support surface height. Defaults to 0.0 for robot actors and 0.75 for ball.",
)

# built-in deformable trampoline options
parser.add_argument("--pin_width", type=float, default=3.0, help="Pinned rim width in meters for the deformable trampoline.")
parser.add_argument("--youngs_modulus", type=float, default=3.0e7, help="Built-in trampoline Young's modulus.")
parser.add_argument("--mass", type=float, default=10.0, help="Built-in trampoline mass.")
parser.add_argument("--sim_resolution", type=int, default=20, help="Built-in trampoline hexahedral resolution.")
parser.add_argument(
    "--randomize_on_reset",
    action="store_true",
    default=False,
    help="Randomize deformable trampoline material and mass on reset.",
)
parser.add_argument(
    "--show_trampoline_nodes",
    action="store_true",
    default=True,
    help="Visualize built-in trampoline nodes, colored by pinned vs. free nodes.",
)
parser.add_argument(
    "--trampoline_node_marker_radius",
    type=float,
    default=0.012,
    help="Marker radius for the optional trampoline node visualization.",
)

# custom spring model options
parser.add_argument("--normal_stiffness", type=float, default=900.0, help="Custom model normal stiffness.")
parser.add_argument("--normal_damping", type=float, default=30.0, help="Custom model normal damping.")
parser.add_argument("--tangential_stiffness", type=float, default=2000.0, help="Custom model tangential stiffness.")
parser.add_argument("--tangential_damping", type=float, default=75.0, help="Custom model tangential damping.")
parser.add_argument("--friction_coeff", type=float, default=1.0, help="Custom model friction coefficient.")
parser.add_argument(
    "--max_force",
    type=float,
    default=None,
    help="Optional force clamp for the ball spring model. Ignored for robot actors.",
)

# ball options
parser.add_argument("--ball_radius", type=float, default=0.18, help="Ball radius in meters.")
parser.add_argument("--ball_mass", type=float, default=6.0, help="Ball mass in kilograms.")
parser.add_argument(
    "--ball_height",
    type=float,
    default=None,
    help="Initial ball center height. Defaults to support height + 0.8m.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.go2 import GO2_CFG
from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_DR_MASS_RANGE,
    TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    TRAMPOLINE_THICKNESS,
    build_trampoline_kinematic_targets,
    make_trampoline_cfg,
    set_trampoline_youngs_moduli,
    trampoline_mesh_prim_path,
)


SPRING_PLANE_RADIUS = 1.5
SPRING_PLANE_THICKNESS = 0.02
G1_CONTACT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
G1_CONTACT_POINT_OFFSETS_LOCAL = ((0.04, 0.0, -0.037), (0.04, 0.0, -0.037))
GO2_CONTACT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
GO2_CONTACT_POINT_OFFSETS_LOCAL = (
    (-0.002, 0.0, -0.022),
    (-0.002, 0.0, -0.022),
    (-0.002, 0.0, -0.022),
    (-0.002, 0.0, -0.022),
)


def make_trampoline_node_marker_cfg(
    prim_path: str,
    color: tuple[float, float, float],
) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "node": sim_utils.SphereCfg(
                radius=args_cli.trampoline_node_marker_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


@configclass
class UnifiedTrampolineSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg | None = None
    ball: RigidObjectCfg | None = None
    trampoline: DeformableObjectCfg | None = None
    spring_visual: AssetBaseCfg | None = None
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


def resolve_surface_height() -> float:
    if args_cli.surface_height is not None:
        return args_cli.surface_height
    return 0.0 if args_cli.actor != "ball" else 0.75


def resolve_ball_height(surface_height: float) -> float:
    if args_cli.ball_height is not None:
        return args_cli.ball_height
    return surface_height + 0.8


def resolve_reset_interval() -> int:
    if args_cli.reset_interval is not None:
        return args_cli.reset_interval
    return 500 if args_cli.actor != "ball" else 360


def resolve_env_spacing() -> float:
    if args_cli.env_spacing is not None:
        return args_cli.env_spacing
    return 4.0 if args_cli.actor != "ball" or args_cli.trampoline_mode == "builtin" else 3.0


def make_ball_cfg(prim_path: str, ball_height: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=args_cli.ball_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7 if args_cli.trampoline_mode == "builtin" else 0.0,
                dynamic_friction=0.6 if args_cli.trampoline_mode == "builtin" else 0.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, ball_height)),
    )


def make_spring_visual_cfg(prim_path: str, surface_height: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCylinderCfg(
            radius=SPRING_PLANE_RADIUS,
            height=SPRING_PLANE_THICKNESS,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), metallic=0.05),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, surface_height - 0.5 * SPRING_PLANE_THICKNESS)),
    )


def build_scene_cfg(surface_height: float, ball_height: float, env_spacing: float) -> UnifiedTrampolineSceneCfg:
    scene_cfg = UnifiedTrampolineSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=env_spacing,
        replicate_physics=args_cli.trampoline_mode != "builtin",
    )
    if args_cli.actor == "g1":
        scene_cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.actor == "go2":
        scene_cfg.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        scene_cfg.ball = make_ball_cfg("{ENV_REGEX_NS}/Ball", ball_height)

    if args_cli.trampoline_mode == "builtin":
        scene_cfg.trampoline = make_trampoline_cfg(
            "{ENV_REGEX_NS}/Trampoline",
            center_z=surface_height - 0.5 * TRAMPOLINE_THICKNESS,
            mass=args_cli.mass,
            youngs_modulus=args_cli.youngs_modulus,
            sim_resolution=args_cli.sim_resolution,
        )
    else:
        scene_cfg.spring_visual = make_spring_visual_cfg("{ENV_REGEX_NS}/SpringPlaneVisual", surface_height)
    return scene_cfg


def set_camera(sim: SimulationContext, surface_height: float) -> None:
    if args_cli.actor == "g1":
        sim.set_camera_view(eye=[2.5, 2.5, surface_height + 1.8], target=[0.0, 0.0, surface_height + 0.7])
    elif args_cli.actor == "go2":
        sim.set_camera_view(eye=[2.2, 2.2, surface_height + 1.2], target=[0.0, 0.0, surface_height + 0.3])
    else:
        sim.set_camera_view(eye=[4.5, -4.0, surface_height + 2.0], target=[0.0, 0.0, surface_height + 0.2])


def clear_ball_wrench(ball: RigidObject) -> None:
    zero_force = torch.zeros((ball.num_instances, ball.num_bodies, 3), device=ball.device)
    zero_torque = torch.zeros_like(zero_force)
    ball.permanent_wrench_composer.set_forces_and_torques(
        forces=zero_force,
        torques=zero_torque,
        is_global=True,
    )


def reset_ball(scene: InteractiveScene, ball: RigidObject) -> None:
    env_ids = torch.arange(scene.num_envs, device=ball.device, dtype=torch.long)
    default_root_state = ball.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    ball.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    ball.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    ball.reset()
    clear_ball_wrench(ball)


def reset_robot(scene: InteractiveScene, robot: Articulation, surface_height: float) -> None:
    env_ids = torch.arange(scene.num_envs, device=robot.device, dtype=torch.long)
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    default_root_state[:, 2] += surface_height
    if args_cli.robot_reset_mode == "drop":
        default_root_state[:, 2] += args_cli.drop_height
        default_root_state[:, 7] = args_cli.drop_vx
        default_root_state[:, 8] = args_cli.drop_vy
        default_root_state[:, 9] = args_cli.drop_vz
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    robot.reset()


def resolve_trampoline_mass_attrs(trampoline: DeformableObject) -> list:
    stage = get_current_stage()
    attrs = []
    for prim_path in sim_utils.find_matching_prim_paths(trampoline_mesh_prim_path(trampoline.cfg.prim_path)):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid trampoline mesh prim: '{prim_path}'.")
        attr = prim.GetAttribute("physics:mass")
        if not attr.IsValid():
            attr = prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float)
        attrs.append(attr)
    return attrs


def apply_deformable_trampoline_properties(
    trampoline: DeformableObject,
    mass_attrs: list,
    env_ids: torch.Tensor,
) -> tuple[float, float]:
    if args_cli.randomize_on_reset:
        youngs_moduli = sample_uniform(
            TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE[0],
            TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE[1],
            (len(env_ids),),
            device=trampoline.device,
        ).to(dtype=torch.float32)
        masses = sample_uniform(
            TRAMPOLINE_DR_MASS_RANGE[0],
            TRAMPOLINE_DR_MASS_RANGE[1],
            (len(env_ids),),
            device=trampoline.device,
        ).to(dtype=torch.float32)
    else:
        youngs_moduli = torch.full(
            (len(env_ids),),
            args_cli.youngs_modulus,
            device=trampoline.device,
            dtype=torch.float32,
        )
        masses = torch.full((len(env_ids),), args_cli.mass, device=trampoline.device, dtype=torch.float32)

    set_trampoline_youngs_moduli(trampoline.material_physx_view, youngs_moduli, env_ids)
    with Sdf.ChangeBlock():
        for env_id, mass in zip(env_ids.tolist(), masses.tolist(), strict=True):
            mass_attrs[env_id].Set(float(mass))
    return float(youngs_moduli[0].item()), float(masses[0].item())


def reset_deformable_trampoline(
    scene: InteractiveScene,
    trampoline: DeformableObject,
    trampoline_targets: torch.Tensor,
    mass_attrs: list,
) -> tuple[float, float]:
    env_ids = torch.arange(scene.num_envs, device=trampoline.device, dtype=torch.long)
    trampoline.write_nodal_state_to_sim(trampoline.data.default_nodal_state_w, env_ids=env_ids)
    youngs_modulus, mass = apply_deformable_trampoline_properties(trampoline, mass_attrs, env_ids)
    trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets, env_ids=env_ids)
    trampoline.reset(env_ids=env_ids)
    return youngs_modulus, mass


def resolve_robot_contact_setup(actor: str) -> tuple[list[str], tuple[tuple[float, float, float], ...]]:
    if actor == "g1":
        return G1_CONTACT_BODY_NAMES, G1_CONTACT_POINT_OFFSETS_LOCAL
    if actor == "go2":
        return GO2_CONTACT_BODY_NAMES, GO2_CONTACT_POINT_OFFSETS_LOCAL
    raise ValueError(f"Unsupported robot actor for spring contact: {actor}.")


def build_robot_spring_runtime(robot: Articulation, actor: str, num_envs: int) -> dict[str, object]:
    from whole_body_tracking.utils.point_foot_contact_force import PointFootContactForceCfg, PointFootContactForceModel

    contact_body_names, contact_point_offsets_local = resolve_robot_contact_setup(actor)
    body_ids, body_names = robot.find_bodies(contact_body_names, preserve_order=True)
    if len(body_ids) != len(contact_body_names):
        raise RuntimeError(f"Failed to resolve custom contact bodies: expected {contact_body_names}, got {body_names}.")

    model = PointFootContactForceModel(
        PointFootContactForceCfg(
            plane_height=resolve_surface_height(),
            normal_stiffness=args_cli.normal_stiffness,
            normal_damping=args_cli.normal_damping,
            tangential_stiffness=args_cli.tangential_stiffness,
            tangential_damping=args_cli.tangential_damping,
            friction_coeff=args_cli.friction_coeff,
        )
    )
    num_bodies = len(body_ids)
    return {
        "model": model,
        "body_ids": list(body_ids),
        "body_names": body_names,
        "offsets_local": torch.tensor(contact_point_offsets_local, device=robot.device, dtype=torch.float32),
        "tangential_displacement_w": torch.zeros((num_envs, num_bodies, 3), device=robot.device),
        "last_penetration": torch.zeros((num_envs, num_bodies), device=robot.device),
        "last_normal_force": torch.zeros((num_envs, num_bodies), device=robot.device),
        "last_tangential_force_norm": torch.zeros((num_envs, num_bodies), device=robot.device),
    }


def clear_robot_spring_contact(robot: Articulation, runtime: dict[str, object]) -> None:
    body_ids = runtime["body_ids"]
    num_bodies = len(body_ids)
    zero_force = torch.zeros((robot.num_instances, num_bodies, 3), device=robot.device)
    zero_torque = torch.zeros_like(zero_force)
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=zero_force,
        torques=zero_torque,
        body_ids=body_ids,
        is_global=True,
    )
    runtime["tangential_displacement_w"].zero_()
    runtime["last_penetration"].zero_()
    runtime["last_normal_force"].zero_()
    runtime["last_tangential_force_norm"].zero_()


def apply_robot_spring_contact(
    scene: InteractiveScene,
    robot: Articulation,
    runtime: dict[str, object],
) -> None:
    body_ids = runtime["body_ids"]
    num_bodies = len(body_ids)
    body_pos_w = robot.data.body_pos_w[:, body_ids]
    body_quat_w = robot.data.body_quat_w[:, body_ids]
    body_lin_vel_w = robot.data.body_lin_vel_w[:, body_ids]
    body_ang_vel_w = robot.data.body_ang_vel_w[:, body_ids]
    env_origins = scene.env_origins.unsqueeze(1).expand(-1, num_bodies, -1)
    offsets_local = runtime["offsets_local"].unsqueeze(0).expand(scene.num_envs, -1, -1)

    force_w, torque_w, _, penetration, normal_force, tangential_force_norm, _, tangential_displacement_w = runtime[
        "model"
    ].compute_wrenches(
        body_pos_w.reshape(-1, 3),
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
        body_ang_vel_w.reshape(-1, 3),
        offsets_local.reshape(-1, 3),
        env_origins=env_origins.reshape(-1, 3),
        tangential_displacement_w=runtime["tangential_displacement_w"].reshape(-1, 3),
        dt=scene.physics_dt,
    )

    force_w = force_w.view(scene.num_envs, num_bodies, 3)
    torque_w = torque_w.view(scene.num_envs, num_bodies, 3)
    runtime["tangential_displacement_w"] = tangential_displacement_w.view(scene.num_envs, num_bodies, 3)
    runtime["last_penetration"] = penetration.view(scene.num_envs, num_bodies)
    runtime["last_normal_force"] = normal_force.view(scene.num_envs, num_bodies)
    runtime["last_tangential_force_norm"] = tangential_force_norm.view(scene.num_envs, num_bodies)

    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=force_w,
        torques=torque_w,
        body_ids=body_ids,
        is_global=True,
    )


def apply_ball_spring_contact(ball: RigidObject, spring_plane: VerticalSpringPlane) -> tuple[torch.Tensor, torch.Tensor]:
    force_w, penetration, _ = spring_plane.compute_force(ball.data.root_pos_w, ball.data.root_lin_vel_w)
    zero_torque = torch.zeros_like(force_w)
    ball.permanent_wrench_composer.set_forces_and_torques(
        forces=force_w.unsqueeze(1),
        torques=zero_torque.unsqueeze(1),
        is_global=True,
    )
    return force_w, penetration


def print_mode_summary(surface_height: float, ball_height: float | None) -> None:
    summary = (
        f"[INFO]: Loaded trampoline scene — actor={args_cli.actor}, mode={args_cli.trampoline_mode}, "
        f"num_envs={args_cli.num_envs}, passive={args_cli.passive}, surface_height={surface_height:.3f}"
    )
    if args_cli.actor == "ball" and ball_height is not None:
        summary += f", ball_height={ball_height:.3f}"
    elif args_cli.actor != "ball":
        summary += f", robot_reset_mode={args_cli.robot_reset_mode}"
        if args_cli.robot_reset_mode == "drop":
            summary += (
                f", drop_height={args_cli.drop_height:.3f}, "
                f"drop_velocity=({args_cli.drop_vx:.3f}, {args_cli.drop_vy:.3f}, {args_cli.drop_vz:.3f})"
            )
    print(summary)
    if args_cli.trampoline_mode == "spring" and args_cli.actor == "g1":
        print("[INFO]: Custom G1 mode uses the point-foot contact approximation on the ankle roll links.")
    if args_cli.trampoline_mode == "spring" and args_cli.actor == "go2":
        print("[INFO]: Custom Go2 mode uses the point-foot contact approximation on the four foot links.")


def build_trampoline_node_visualizers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    pinned_visualizer = VisualizationMarkers(
        make_trampoline_node_marker_cfg("/Visuals/TrampolinePinnedNodes", color=(1.0, 0.2, 0.2))
    )
    free_visualizer = VisualizationMarkers(
        make_trampoline_node_marker_cfg("/Visuals/TrampolineFreeNodes", color=(0.2, 1.0, 0.2))
    )
    return pinned_visualizer, free_visualizer


def update_trampoline_node_visualizers(
    trampoline: DeformableObject,
    pinned_mask: torch.Tensor,
    pinned_visualizer: VisualizationMarkers | None,
    free_visualizer: VisualizationMarkers | None,
) -> None:
    if pinned_visualizer is None or free_visualizer is None:
        return

    env0_mask = pinned_mask[0]
    env0_nodal_pos_w = trampoline.data.nodal_pos_w[0]
    pinned_visualizer.visualize(translations=env0_nodal_pos_w[env0_mask])
    free_visualizer.visualize(translations=env0_nodal_pos_w[~env0_mask])


def main() -> None:
    surface_height = resolve_surface_height()
    ball_height = resolve_ball_height(surface_height) if args_cli.actor == "ball" else None
    env_spacing = resolve_env_spacing()
    reset_interval = resolve_reset_interval()

    sim_dt = 0.005 if args_cli.actor != "ball" else 1.0 / 120.0
    sim = SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device))
    set_camera(sim, surface_height)

    scene_cfg = build_scene_cfg(surface_height, ball_height if ball_height is not None else 0.0, env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(sim_dt)

    actor = scene["robot"] if args_cli.actor != "ball" else scene["ball"]

    trampoline = None
    trampoline_targets = None
    trampoline_pinned_mask = None
    trampoline_center_node_ids = None
    trampoline_mass_attrs = None
    current_youngs_modulus = None
    current_mass = None
    trampoline_pinned_node_visualizer = None
    trampoline_free_node_visualizer = None
    if args_cli.trampoline_mode == "builtin":
        trampoline = scene["trampoline"]
        if trampoline.material_physx_view is None:
            raise RuntimeError("Failed to create deformable trampoline material view.")
        trampoline_targets, trampoline_pinned_mask, trampoline_center_node_ids = build_trampoline_kinematic_targets(
            trampoline.data.default_nodal_state_w,
            trampoline.data.nodal_kinematic_target,
            pin_width=args_cli.pin_width,
        )
        trampoline_mass_attrs = resolve_trampoline_mass_attrs(trampoline)
        if args_cli.show_trampoline_nodes:
            trampoline_pinned_node_visualizer, trampoline_free_node_visualizer = build_trampoline_node_visualizers()
    elif args_cli.show_trampoline_nodes:
        print("[INFO]: `--show_trampoline_nodes` only applies to `--trampoline_mode builtin`; ignoring it in spring mode.")

    spring_ball = None
    spring_robot_runtime = None
    if args_cli.trampoline_mode == "spring":
        if args_cli.actor == "ball":
            from whole_body_tracking.utils.spring_terrain import VerticalSpringPlane, VerticalSpringPlaneCfg

            spring_ball = VerticalSpringPlane(
                VerticalSpringPlaneCfg(
                    plane_height=surface_height,
                    stiffness=args_cli.normal_stiffness,
                    damping=args_cli.normal_damping,
                    contact_radius=args_cli.ball_radius,
                    max_force=args_cli.max_force,
                )
            )
        else:
            spring_robot_runtime = build_robot_spring_runtime(actor, args_cli.actor, scene.num_envs)
            print(f"[INFO]: Resolved custom contact bodies: {spring_robot_runtime['body_names']}")

    if args_cli.actor == "ball":
        reset_ball(scene, actor)
    else:
        reset_robot(scene, actor, surface_height)
        if spring_robot_runtime is not None:
            clear_robot_spring_contact(actor, spring_robot_runtime)

    if trampoline is not None:
        current_youngs_modulus, current_mass = reset_deformable_trampoline(
            scene,
            trampoline,
            trampoline_targets,
            trampoline_mass_attrs,
        )

    print_mode_summary(surface_height, ball_height)
    if current_youngs_modulus is not None and current_mass is not None:
        print(
            f"[INFO]: Built-in trampoline properties — youngs_modulus={current_youngs_modulus:.1f}, mass={current_mass:.3f}"
        )
    if trampoline_pinned_mask is not None:
        pinned_node_count = int(trampoline_pinned_mask[0].sum().item())
        free_node_count = int((~trampoline_pinned_mask[0]).sum().item())
        print(f"[INFO]: Trampoline nodes — pinned={pinned_node_count}, free={free_node_count}")
        if trampoline_pinned_node_visualizer is not None and trampoline_free_node_visualizer is not None:
            update_trampoline_node_visualizers(
                trampoline,
                trampoline_pinned_mask,
                trampoline_pinned_node_visualizer,
                trampoline_free_node_visualizer,
            )

    step = 0
    while simulation_app.is_running():
        if step > 0 and step % reset_interval == 0:
            if args_cli.actor == "ball":
                reset_ball(scene, actor)
            else:
                reset_robot(scene, actor, surface_height)
                if spring_robot_runtime is not None:
                    clear_robot_spring_contact(actor, spring_robot_runtime)
            if trampoline is not None:
                current_youngs_modulus, current_mass = reset_deformable_trampoline(
                    scene,
                    trampoline,
                    trampoline_targets,
                    trampoline_mass_attrs,
                )
                print(
                    f"[INFO]: Resetting built-in trampoline — youngs_modulus={current_youngs_modulus:.1f}, mass={current_mass:.3f}"
                )
            else:
                print("[INFO]: Resetting spring trampoline scene.")

        if trampoline is not None:
            trampoline.write_nodal_kinematic_target_to_sim(trampoline_targets)
        elif spring_ball is not None:
            spring_force_w, spring_penetration = apply_ball_spring_contact(actor, spring_ball)
        elif spring_robot_runtime is not None:
            apply_robot_spring_contact(scene, actor, spring_robot_runtime)

        if args_cli.actor != "ball" and not args_cli.passive:
            actor.set_joint_position_target(actor.data.default_joint_pos.clone())

        actor.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if trampoline is not None and trampoline_pinned_mask is not None:
            update_trampoline_node_visualizers(
                trampoline,
                trampoline_pinned_mask,
                trampoline_pinned_node_visualizer,
                trampoline_free_node_visualizer,
            )

        if step % args_cli.print_interval == 0:
            if args_cli.actor == "ball" and trampoline is not None:
                center_height = trampoline.data.nodal_pos_w[0, trampoline_center_node_ids[0], 2].item()
                print(
                    f"[INFO]: step={step}, ball_z={actor.data.root_pos_w[0, 2].item():.3f}, "
                    f"trampoline_center_z={center_height:.3f}"
                )
            elif args_cli.actor == "ball":
                print(
                    f"[INFO]: step={step}, ball_z={actor.data.root_pos_w[0, 2].item():.3f}, "
                    f"penetration={spring_penetration[0].item():.4f}, force_z={spring_force_w[0, 2].item():.3f}"
                )
            elif trampoline is not None:
                center_height = trampoline.data.nodal_pos_w[0, trampoline_center_node_ids[0], 2].item()
                print(
                    f"[INFO]: step={step}, root_z={actor.data.root_pos_w[0, 2].item():.3f}, "
                    f"trampoline_center_z={center_height:.3f}"
                )
            else:
                mean_penetration = spring_robot_runtime["last_penetration"][0].mean().item()
                mean_normal_force = spring_robot_runtime["last_normal_force"][0].mean().item()
                print(
                    f"[INFO]: step={step}, root_z={actor.data.root_pos_w[0, 2].item():.3f}, "
                    f"mean_foot_penetration={mean_penetration:.4f}, mean_foot_normal_force={mean_normal_force:.3f}"
                )

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()