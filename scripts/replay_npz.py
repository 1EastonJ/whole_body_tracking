"""Replay a converted motion npz for G1 or Go2 inside IsaacLab.

Examples:
    python scripts/replay_npz.py --motion_file artifacts/go2_frontflip:v0/motion.npz --actor go2
    python scripts/replay_npz.py --registry_name my-org/wandb-registry-motions/my_motion:latest --actor g1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a converted motion npz for G1 or Go2.")
parser.add_argument("--motion_file", type=str, help="Path to a local motion .npz file.")
parser.add_argument("--registry_name", type=str, help="Optional wandb registry motion reference.")
parser.add_argument("--actor", type=str, choices=("g1", "go2"), default="g1", help="Robot to replay.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if bool(args_cli.motion_file) == bool(args_cli.registry_name):
    raise ValueError("Provide exactly one of --motion_file or --registry_name.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.go2 import GO2_CFG


class NpzMotion:
    def __init__(self, motion_file: str, device: str):
        data = np.load(motion_file)
        self.fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.time_step_total = self.joint_pos.shape[0]


def resolve_motion_file() -> str:
    if args_cli.motion_file:
        motion_path = pathlib.Path(args_cli.motion_file).expanduser().resolve()
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        return str(motion_path)

    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    return str(pathlib.Path(artifact.download()) / "motion.npz")


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion: NpzMotion):
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if robot.num_joints != motion.joint_pos.shape[1]:
        raise RuntimeError(
            f"Joint count mismatch between actor and motion: robot has {robot.num_joints} joints, "
            f"but motion contains {motion.joint_pos.shape[1]} joints."
        )
    if motion.body_pos_w.shape[1] == 0:
        raise RuntimeError("Motion file does not contain any body states.")

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    while simulation_app.is_running():
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps, 0]
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion.body_quat_w[time_steps, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        time_steps += 1


def main():
    motion_file = resolve_motion_file()
    motion_meta = NpzMotion(motion_file, device="cpu")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(motion_meta.fps)
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    if args_cli.actor == "go2":
        scene_cfg.robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        scene_cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print(f"[INFO]: Setup complete. Replaying {motion_file} with actor={args_cli.actor}, fps={motion_meta.fps}")

    motion = NpzMotion(motion_file, device=sim.device)
    run_simulator(sim, scene, motion)


if __name__ == "__main__":
    main()
    simulation_app.close()
