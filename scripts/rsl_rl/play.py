"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata
import os
import pathlib
import sys


def _add_repo_source_to_path():
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / "source" / "whole_body_tracking"
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_add_repo_source_to_path()

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint file. Can be absolute or relative to the workspace.",
)
parser.add_argument("--command_vx", type=float, default=None, help="Fixed commanded forward velocity for hopping play.")
parser.add_argument("--command_vy", type=float, default=None, help="Fixed commanded lateral velocity for hopping play.")
parser.add_argument("--command_yaw", type=float, default=None, help="Fixed commanded yaw rate for hopping play.")
parser.add_argument(
    "--torque_stats",
    action="store_true",
    default=False,
    help="Print computed/applied joint torque diagnostics while playing.",
)
parser.add_argument(
    "--torque_stats_interval",
    type=int,
    default=50,
    help="Environment step interval for printing torque diagnostics during play.",
)
parser.add_argument(
    "--torque_stats_top_k",
    type=int,
    default=4,
    help="How many joints to include in each printed torque diagnostic snapshot.",
)
parser.add_argument(
    "--torque_env_id",
    type=int,
    default=0,
    help="Environment index to inspect when printing torque diagnostics.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx, export_policy_as_onnx
from whole_body_tracking.utils.task_utils import env_cfg_requires_motion, env_requires_motion


def _get_rsl_rl_version() -> str:
    try:
        return importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError:
        return importlib.metadata.version("rsl-rl")


def _download_wandb_checkpoint(wandb_path: str) -> tuple[str, str, object]:
    import wandb

    run_path = wandb_path
    api = wandb.Api()
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)
    files = [file.name for file in wandb_run.files() if "model" in file.name]
    if "model" in wandb_path:
        file = wandb_path.split("/")[-1]
    else:
        file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    wandb_file = wandb_run.file(str(file))
    wandb_file.download("./logs/rsl_rl/temp", replace=True)

    print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
    resume_path = f"./logs/rsl_rl/temp/{file}"
    return resume_path, run_path, wandb_run


def _configure_manual_command_for_play(env) -> None:
    if all(value is None for value in (args_cli.command_vx, args_cli.command_vy, args_cli.command_yaw)):
        return

    vx = 0.0 if args_cli.command_vx is None else args_cli.command_vx
    vy = 0.0 if args_cli.command_vy is None else args_cli.command_vy
    yaw = 0.0 if args_cli.command_yaw is None else args_cli.command_yaw

    if hasattr(env.unwrapped, "set_manual_command"):
        env.unwrapped.set_manual_command(vx, vy, yaw)
        print(f"[INFO]: Using fixed hopping command vx={vx:.3f}, vy={vy:.3f}, yaw={yaw:.3f}")
    else:
        print("[INFO]: Ignoring fixed command override because this task does not support manual play commands.")


def _configure_motion_source_for_play(env_cfg, motion_file: str | None, wandb_run) -> None:
    if not env_cfg_requires_motion(env_cfg):
        if motion_file is not None:
            print("[INFO]: Ignoring --motion_file for non-motion task.")
        return

    if motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {motion_file}")
        env_cfg.commands.motion.motion_file = motion_file
        return

    if wandb_run is not None:
        art = next((artifact for artifact in wandb_run.used_artifacts() if artifact.type == "motions"), None)
        if art is None:
            raise ValueError("No motion artifact found in the specified W&B run.")
        env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
        return

    raise ValueError("--motion_file is required when playing a motion-based task from a local checkpoint.")


def _is_full_slice(value) -> bool:
    return isinstance(value, slice) and value.start is None and value.stop is None and value.step is None


def _joint_indices_to_tensor(joint_indices, num_joints: int, device: torch.device) -> torch.Tensor:
    if _is_full_slice(joint_indices):
        return torch.arange(num_joints, device=device, dtype=torch.long)
    if isinstance(joint_indices, torch.Tensor):
        return joint_indices.to(device=device, dtype=torch.long)
    return torch.as_tensor(joint_indices, device=device, dtype=torch.long)


def _build_joint_effort_limits(robot) -> torch.Tensor:
    effort_limits = robot.data.joint_effort_limits[0].clone()
    for actuator in robot.actuators.values():
        joint_indices = _joint_indices_to_tensor(actuator.joint_indices, robot.num_joints, robot.device)
        effort_limits[joint_indices] = actuator.effort_limit[0]
    return torch.clamp(effort_limits, min=1.0e-6)


def _print_torque_stats(robot, effort_limits: torch.Tensor, play_step: int) -> None:
    num_envs = robot.data.applied_torque.shape[0]
    env_id = max(0, min(args_cli.torque_env_id, num_envs - 1))

    computed = robot.data.computed_torque[env_id]
    applied = robot.data.applied_torque[env_id]
    joint_vel = robot.data.joint_vel[env_id]

    computed_ratio = computed.abs() / effort_limits
    applied_ratio = applied.abs() / effort_limits
    clip_ratio = (computed - applied).abs() / effort_limits
    top_metric = torch.maximum(computed_ratio, applied_ratio)
    top_k = max(1, min(args_cli.torque_stats_top_k, top_metric.numel()))
    _, top_indices = torch.topk(top_metric, k=top_k)
    saturated = int((clip_ratio > 1.0e-3).sum().item())

    print(
        f"[TORQUE] step={play_step} env={env_id} "
        f"max|computed|/limit={computed_ratio.max().item():.3f} "
        f"max|applied|/limit={applied_ratio.max().item():.3f} "
        f"saturated_joints={saturated}/{top_metric.numel()}"
    )
    for idx in top_indices.tolist():
        print(
            f"  {robot.data.joint_names[idx]}: "
            f"comp={computed[idx].item():.2f}, "
            f"app={applied[idx].item():.2f}, "
            f"limit={effort_limits[idx].item():.2f}, "
            f"vel={joint_vel[idx].item():.2f}, "
            f"comp/lim={computed_ratio[idx].item():.3f}, "
            f"app/lim={applied_ratio[idx].item():.3f}, "
            f"clip/lim={clip_ratio[idx].item():.3f}"
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    rsl_rl_version = _get_rsl_rl_version()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    wandb_run = None
    run_path_for_metadata = "none"
    if args_cli.wandb_path:
        resume_path, run_path_for_metadata, wandb_run = _download_wandb_checkpoint(args_cli.wandb_path)
    elif args_cli.checkpoint_path:
        resume_path = os.path.abspath(args_cli.checkpoint_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        run_path_for_metadata = resume_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    _configure_motion_source_for_play(env_cfg, args_cli.motion_file, wandb_run)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, rsl_rl_version)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped
    _configure_manual_command_for_play(env)
    log_dir = os.path.dirname(resume_path)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    torque_effort_limits = None
    if args_cli.torque_stats:
        torque_effort_limits = _build_joint_effort_limits(base_env.scene["robot"])
        print(
            f"[INFO]: Torque diagnostics enabled for env {args_cli.torque_env_id} "
            f"every {max(args_cli.torque_stats_interval, 1)} play steps."
        )

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(
        resume_path,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": True, "rnd": False},
        strict=False,
    )

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    obs_normalizer = getattr(ppo_runner, "obs_normalizer", None)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if hasattr(ppo_runner.alg, "policy"):
        if env_requires_motion(env.unwrapped):
            export_motion_policy_as_onnx(
                env.unwrapped,
                ppo_runner.alg.policy,
                normalizer=obs_normalizer,
                path=export_model_dir,
                filename="policy.onnx",
            )
        else:
            export_policy_as_onnx(
                ppo_runner.alg.policy,
                normalizer=obs_normalizer,
                path=export_model_dir,
                filename="policy.onnx",
            )
        attach_onnx_metadata(env.unwrapped, run_path_for_metadata, export_model_dir)
    else:
        print("[INFO]: Skipping legacy ONNX export because this rsl-rl version uses separate actor/critic models.")

    obs_output = env.get_observations()
    obs = obs_output[0] if isinstance(obs_output, tuple) else obs_output
    play_step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            step_output = env.step(actions)
            obs = step_output[0] if isinstance(step_output, tuple) else step_output

        play_step += 1
        if args_cli.torque_stats and play_step % max(args_cli.torque_stats_interval, 1) == 0:
            _print_torque_stats(base_env.scene["robot"], torque_effort_limits, play_step)

        if args_cli.video and play_step == args_cli.video_length:
            break

    env.close()


if __name__ == "__main__":
    if args_cli.task == "Hopping-Trampoline-Go2-TrackingCfg-v0":
        os.environ.pop("WHOLE_BODY_TRACKING_PLAY_MODE", None)
        print("[INFO]: Playing Hopping-Trampoline-Go2-TrackingCfg-v0 with training environment settings.")
    else:
        os.environ["WHOLE_BODY_TRACKING_PLAY_MODE"] = "1"
    main()
    simulation_app.close()
