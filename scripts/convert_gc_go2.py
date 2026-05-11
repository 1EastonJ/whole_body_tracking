"""Convert a Go2 generalized-coordinate CSV motion into IsaacLab motion .npz format.

Example:
    python scripts/convert_gc_go2.py --input_file /path/to/go2_motion.csv --input_fps 30 \
        --output_name my_go2_motion --output_fps 50
"""

from __future__ import annotations

# Launch Isaac Sim Simulator first.

import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a Go2 motion CSV and export it as a tracking .npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input Go2 motion CSV file.")
parser.add_argument("--input_fps", type=int, default=50, help="The FPS of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "Frame range: START END (both inclusive). The frame index starts from 1 over data rows only."
        " If not provided, all frames are loaded."
    ),
)
parser.add_argument("--output_name", type=str, required=True, help="The wandb registry name for the motion npz.")
parser.add_argument("--output_fps", type=int, default=50, help="The FPS of the output motion.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest everything follows.

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.go2 import GO2_CFG, GO2_CSV_JOINT_NAMES

EXPECTED_COLUMNS = 19
# CSV columns 7:19 are in mjlab Go2 order. We place them into IsaacLab by resolving the same joint names.
GO2_INPUT_JOINT_NAMES = list(GO2_CSV_JOINT_NAMES)


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

    robot: ArticulationCfg = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    @staticmethod
    def _has_header(csv_path: str) -> bool:
        with open(csv_path, encoding="utf-8") as file:
            first_line = file.readline().strip()
        if not first_line:
            return False
        try:
            [float(value) for value in first_line.split(",")]
            return False
        except ValueError:
            return True

    def _load_csv(self) -> torch.Tensor:
        header_rows = 1 if self._has_header(self.motion_file) else 0
        if self.frame_range is None:
            data = np.loadtxt(self.motion_file, delimiter=",", skiprows=header_rows)
        else:
            data = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=header_rows + self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
            )
        tensor = torch.from_numpy(data).to(torch.float32).to(self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _load_motion(self):
        motion = self._load_csv()
        if motion.shape[1] != EXPECTED_COLUMNS:
            raise ValueError(
                f"Go2 CSV must have {EXPECTED_COLUMNS} columns (3 root pos + 4 root quat + 12 joints),"
                f" but got {motion.shape[1]}."
            )
        if motion.shape[0] < 2:
            raise ValueError(f"Go2 CSV must contain at least 2 frames, but got {motion.shape[0]}.")

        self.motion_base_poss_input = motion[:, :3]
        quat_xyzw = motion[:, 3:7]
        self.motion_base_rots_input = quat_xyzw[:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration:.3f} sec,"
            f" frames: {self.input_frames}"
        )

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps},"
            f" output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for index in range(a.shape[0]):
            slerped_quats[index] = quat_slerp(a[index], b[index], float(blend[index]))
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(
            index_0 + 1,
            torch.tensor(self.input_frames - 1, device=self.device, dtype=torch.long),
        )
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        bool,
    ]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    robot_joint_indexes, resolved_joint_names = robot.find_joints(joint_names, preserve_order=True)
    if len(robot_joint_indexes) != len(joint_names):
        raise RuntimeError(
            f"Failed to resolve all Go2 joints in IsaacLab. Expected {len(joint_names)} joints,"
            f" got {len(robot_joint_indexes)}."
        )
    if list(resolved_joint_names) != list(joint_names):
        raise RuntimeError(
            "Resolved Go2 joint names do not match the expected mjlab CSV order. "
            f"Expected {joint_names}, got {list(resolved_joint_names)}."
        )
    if motion.motion_dof_poss_input.shape[1] != len(joint_names):
        raise RuntimeError(
            "Go2 CSV DOF count does not match the expected joint list length. "
            f"Expected {len(joint_names)}, got {motion.motion_dof_poss_input.shape[1]}."
        )

    print("[INFO]: Go2 CSV column to IsaacLab joint mapping:")
    for csv_column, (joint_name, joint_index) in enumerate(zip(joint_names, robot_joint_indexes), start=7):
        print(f"  csv[{csv_column}] -> {joint_name} -> IsaacLab joint id {joint_index}")

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag:
            for key in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[key] = np.stack(log[key], axis=0)

            np.savez("/tmp/motion.npz", **log)

            import wandb

            collection = args_cli.output_name
            run = wandb.init(project="csv_to_npz", name=collection)
            print(f"[INFO]: Logging motion to wandb: {collection}")
            registry = "motions"
            logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=collection, type=registry)
            run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{registry}/{collection}")
            print(f"[INFO]: Motion saved to wandb registry: {registry}/{collection}")
            break


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene, joint_names=GO2_INPUT_JOINT_NAMES)


if __name__ == "__main__":
    main()
    simulation_app.close()
