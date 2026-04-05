from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp.events import randomize_actuator_gains, randomize_rigid_body_mass
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

from whole_body_tracking.robots.go2 import GO2_CSV_JOINT_NAMES
from whole_body_tracking.tasks.hopping.config.go2.flat_env_cfg import Go2HoppingFlatEnvCfg
from whole_body_tracking.tasks.hopping.symmetry import GO2_JUMP_ACTION_PERMUTATION, GO2_JUMP_OBS_PERMUTATION
from whole_body_tracking.tasks.tracking.mdp.events import randomize_rigid_body_com
from whole_body_tracking.utils.trampoline_deformable import (
    build_trampoline_kinematic_targets,
    build_trampoline_visual_translate_ops,
    reset_deformable_trampoline,
    trampoline_center_heights,
    update_trampoline_visual_height,
)


def _invoke_randomization_term(term, env, env_ids, **kwargs):
    if isinstance(term, type) and issubclass(term, ManagerTermBase):
        cfg = EventTermCfg(func=term, params=kwargs)
        term(cfg, env)(env, env_ids, **kwargs)
    else:
        term(env, env_ids, **kwargs)


class Go2HoppingEnv(DirectRLEnv):
    cfg: Go2HoppingFlatEnvCfg

    def __init__(self, cfg: Go2HoppingFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._canonical_joint_names = list(GO2_CSV_JOINT_NAMES)
        self._canonical_joint_ids = self._to_tensor(
            self._robot.find_joints(self._canonical_joint_names, preserve_order=True)[0]
        )
        resolved_joint_names = [self._robot.data.joint_names[idx] for idx in self._canonical_joint_ids.tolist()]
        if resolved_joint_names != self._canonical_joint_names:
            raise RuntimeError(
                f"Go2 hopping joint order mismatch. Expected {self._canonical_joint_names}, got {resolved_joint_names}."
            )

        self.num_states = gym.spaces.flatdim(self.single_observation_space["critic"])
        self._action_dim = len(self._canonical_joint_names)
        if self._action_dim != gym.spaces.flatdim(self.single_action_space):
            raise RuntimeError("Configured action space does not match canonical Go2 joint order length.")
        self._policy_frame_dim = 47
        self._critic_frame_dim = 70
        self._policy_frame_stack = 10
        self._critic_frame_stack = 3
        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self._command_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ],
            device=self.device,
            dtype=torch.float,
        )
        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float).view(1, -1)
        if self._action_scale.shape[-1] != self._action_dim:
            raise RuntimeError("Configured action scale does not match canonical Go2 joint order length.")
        self._command_xy_deadzone = float(self.cfg.commands.command_xy_deadzone)
        self._has_trampoline = self.cfg.trampoline is not None
        self._foot_body_names = tuple(self.cfg.foot_body_names)
        self._penalized_contact_body_names = tuple(self.cfg.penalized_contact_body_names)
        self._termination_body_names = tuple(self.cfg.termination_body_names)
        self._non_foot_contact_body_names = tuple(self.cfg.non_foot_contact_body_names)
        self._reward_scales = OrderedDict(
            (
                ("termination", self.cfg.rewards.scales.termination * self.step_dt),
                ("tracking_lin_vel", self.cfg.rewards.scales.tracking_lin_vel * self.step_dt),
                ("tracking_ang_vel", self.cfg.rewards.scales.tracking_ang_vel * self.step_dt),
                ("lin_vel_z", self.cfg.rewards.scales.lin_vel_z * self.step_dt),
                ("ang_vel_xy", self.cfg.rewards.scales.ang_vel_xy * self.step_dt),
                ("orientation", self.cfg.rewards.scales.orientation * self.step_dt),
                ("torques", self.cfg.rewards.scales.torques * self.step_dt),
                ("dof_vel", self.cfg.rewards.scales.dof_vel * self.step_dt),
                ("dof_acc", self.cfg.rewards.scales.dof_acc * self.step_dt),
                ("base_height", self.cfg.rewards.scales.base_height * self.step_dt),
                ("feet_air_time", self.cfg.rewards.scales.feet_air_time * self.step_dt),
                ("collision", self.cfg.rewards.scales.collision * self.step_dt),
                ("feet_stumble", self.cfg.rewards.scales.feet_stumble * self.step_dt),
                ("action_rate", self.cfg.rewards.scales.action_rate * self.step_dt),
                ("default_pos", self.cfg.rewards.scales.default_pos * self.step_dt),
                ("default_hip_pos", self.cfg.rewards.scales.default_hip_pos * self.step_dt),
                ("feet_contact_forces", self.cfg.rewards.scales.feet_contact_forces * self.step_dt),
                ("jump", self.cfg.rewards.scales.jump * self.step_dt),
                ("feet_clearance", self.cfg.rewards.scales.feet_clearance * self.step_dt),
            )
        )
        self._reward_names = [
            name for name, scale in self._reward_scales.items() if name != "termination" and scale != 0.0
        ]
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self._reward_scales
        }

        self.actions = torch.zeros(self.num_envs, self._action_dim, dtype=torch.float, device=self.device)
        self._last_actions = torch.zeros_like(self.actions)
        self._last_dof_vel = torch.zeros(self.num_envs, self._action_dim, dtype=torch.float, device=self.device)
        self._commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device)
        self._manual_command: torch.Tensor | None = None
        self._feet_air_time = torch.zeros(
            self.num_envs, len(self._foot_body_names), dtype=torch.float, device=self.device
        )
        self._last_contacts = torch.zeros(
            self.num_envs, len(self._foot_body_names), dtype=torch.bool, device=self.device
        )
        self._env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self._body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self._motor_zero_offsets = torch.zeros(self.num_envs, self._action_dim, dtype=torch.float, device=self.device)
        self._obs_history = torch.zeros(
            self.num_envs, self._policy_frame_stack, self._policy_frame_dim, dtype=torch.float, device=self.device
        )
        self._critic_history = torch.zeros(
            self.num_envs, self._critic_frame_stack, self._critic_frame_dim, dtype=torch.float, device=self.device
        )

        max_action_delay = self.cfg.domain_rand.range_cmd_action_latency[1]
        max_motor_delay = self.cfg.domain_rand.range_obs_motor_latency[1]
        max_imu_delay = self.cfg.domain_rand.range_obs_imu_latency[1]
        self._cmd_action_latency_buffer = torch.zeros(
            self.num_envs, self._action_dim, max_action_delay + 1, dtype=torch.float, device=self.device
        )
        self._obs_motor_latency_buffer = torch.zeros(
            self.num_envs, self._action_dim * 2, max_motor_delay + 1, dtype=torch.float, device=self.device
        )
        self._obs_imu_latency_buffer = torch.zeros(
            self.num_envs, 6, max_imu_delay + 1, dtype=torch.float, device=self.device
        )
        self._cmd_action_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs_motor_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs_imu_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._base_body_ids = self._to_tensor(
            self._robot.find_bodies(list(self._termination_body_names), preserve_order=True)[0]
        )
        self._foot_body_ids = self._to_tensor(self._robot.find_bodies(list(self._foot_body_names), preserve_order=True)[0])
        self._contact_base_body_ids = self._to_tensor(
            self._contact_sensor.find_bodies(list(self._termination_body_names), preserve_order=True)[0]
        )
        self._contact_foot_body_ids = self._to_tensor(
            self._contact_sensor.find_bodies(list(self._foot_body_names), preserve_order=True)[0]
        )
        self._contact_penalized_body_ids = self._to_tensor(
            self._contact_sensor.find_bodies(list(self._penalized_contact_body_names), preserve_order=True)[0]
        )
        self._contact_non_foot_body_ids = self._to_tensor(
            self._contact_sensor.find_bodies(list(self._non_foot_contact_body_names), preserve_order=True)[0]
        )
        self._non_base_body_ids = self._to_tensor(
            [body_id for body_id in range(self._robot.num_bodies) if body_id not in self._base_body_ids.tolist()]
        )
        self._term_base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_back_lie = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_out_of_trampoline = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_bad_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._trampoline_targets: torch.Tensor | None = None
        self._trampoline_pinned_mask: torch.Tensor | None = None
        self._trampoline_center_node_ids: torch.Tensor | None = None
        self._trampoline_visual_translate_ops = None

        self._default_dof_pos = self._robot.data.default_joint_pos[:, self._canonical_joint_ids].clone()
        self._default_joint_pd_target = self._default_dof_pos.clone()
        self._noise_scale_vec = self._build_noise_scale_vec()
        self._measured_heights = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self._policy_obs_names = self._build_policy_obs_names()
        self._critic_obs_names = self._build_critic_obs_names()
        self._command_resample_interval = int(self.cfg.commands.resampling_time / self.step_dt)
        self._push_interval = math.ceil(self.cfg.domain_rand.push_interval_s / self.step_dt)

        if self._has_trampoline:
            self._trampoline_targets, self._trampoline_pinned_mask, self._trampoline_center_node_ids = build_trampoline_kinematic_targets(
                self._trampoline.data.default_nodal_state_w,
                self._trampoline.data.nodal_kinematic_target,
                pin_width=self.cfg.trampoline_pin_width,
            )
            reset_deformable_trampoline(self._trampoline, self._trampoline_targets)

        self._apply_startup_randomization(self._all_env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self._update_state_buffers()

    def _setup_scene(self):
        self._terrain = None
        self._trampoline = None

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        if self.cfg.terrain is not None:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        if self.cfg.trampoline is not None:
            self._trampoline = DeformableObject(self.cfg.trampoline)
            self.scene.deformable_objects["trampoline"] = self._trampoline

        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.trampoline is not None and self.cfg.use_plain_trampoline_visual:
            visual_thickness = max(0.01, 0.25 * float(self.cfg.trampoline_thickness))
            visual_cfg = sim_utils.CylinderCfg(
                radius=float(self.cfg.trampoline_radius),
                height=visual_thickness,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.55, 0.55),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=1.0,
                    metallic=0.0,
                ),
            )
            visual_cfg.func(
                "/World/envs/env_.*/TrampolineVisual",
                visual_cfg,
                translation=(0.0, 0.0, float(self.cfg.trampoline_surface_height) - 0.5 * visual_thickness),
            )
            self._trampoline_visual_translate_ops = build_trampoline_visual_translate_ops(self.scene.cfg.num_envs)
        if not self.scene.cfg.replicate_physics:
            self.scene.filter_collisions()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clamp(actions, -clip_actions, clip_actions)

    def _apply_action(self):
        if self.cfg.domain_rand.add_cmd_action_latency:
            self._cmd_action_latency_buffer[:, :, 1:] = self._cmd_action_latency_buffer[:, :, :-1]
            self._cmd_action_latency_buffer[:, :, 0] = self.actions * self._action_scale
            delayed_actions = torch.gather(
                self._cmd_action_latency_buffer,
                2,
                self._cmd_action_latency_simstep[:, None, None].expand(-1, self._action_dim, 1),
            ).squeeze(-1)
        else:
            delayed_actions = self.actions * self._action_scale
        joint_targets = delayed_actions + self._default_dof_pos + self._motor_zero_offsets
        self._robot.set_joint_position_target(joint_targets, joint_ids=self._canonical_joint_ids)
        if self._has_trampoline and self._trampoline_targets is not None:
            self._trampoline.write_nodal_kinematic_target_to_sim(self._trampoline_targets)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        self.extras = {}
        self._reset_idx(self._all_env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self._update_state_buffers()
        observations = self._get_observations()
        self._last_actions.copy_(self.actions)
        self._last_dof_vel.copy_(self._joint_vel())
        return observations, self.extras

    def step(self, action: torch.Tensor):
        self.extras = {}
        action = action.to(self.device)
        self._pre_physics_step(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._update_obs_latency_buffers()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._update_state_buffers()
        self._post_physics_step_callback()

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=self.physics_dt)
            self._update_state_buffers()

        self.obs_buf = self._get_observations()
        self._last_actions.copy_(self.actions)
        self._last_dof_vel.copy_(self._joint_vel())
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _get_observations(self) -> dict[str, torch.Tensor]:
        phase = self._get_phase()
        sin_pos = torch.sin(2.0 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2.0 * torch.pi * phase).unsqueeze(1)
        stance_mask = self._get_gait_phase()
        contact_mask = (self._contact_forces[:, self._contact_foot_body_ids, 2] > 5.0).float()
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()

        command_input = torch.cat((sin_pos, cos_pos, self._commands[:, :3] * self._command_scale), dim=1)
        privileged_frame = torch.cat(
            (
                command_input,
                (joint_pos - self._default_joint_pd_target) * self.cfg.normalization.obs_scales.dof_pos,
                joint_pos * self.cfg.normalization.obs_scales.dof_pos,
                joint_vel * self.cfg.normalization.obs_scales.dof_vel,
                self.actions,
                self._robot.data.root_lin_vel_b * self.cfg.normalization.obs_scales.lin_vel,
                self._robot.data.root_ang_vel_b * self.cfg.normalization.obs_scales.ang_vel,
                self._base_euler_xyz * self.cfg.normalization.obs_scales.quat,
                self._env_frictions,
                self._body_mass / 10.0,
                stance_mask,
                contact_mask,
            ),
            dim=-1,
        )

        q = (joint_pos - self._default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = joint_vel * self.cfg.normalization.obs_scales.dof_vel
        if self.cfg.domain_rand.add_obs_latency and self.cfg.domain_rand.randomize_obs_motor_latency:
            obs_motor = torch.gather(
                self._obs_motor_latency_buffer,
                2,
                self._obs_motor_latency_simstep[:, None, None].expand(-1, self._action_dim * 2, 1),
            ).squeeze(-1)
        else:
            obs_motor = torch.cat((q, dq), dim=1)
        if self.cfg.domain_rand.add_obs_latency and self.cfg.domain_rand.randomize_obs_imu_latency:
            obs_imu = torch.gather(
                self._obs_imu_latency_buffer,
                2,
                self._obs_imu_latency_simstep[:, None, None].expand(-1, 6, 1),
            ).squeeze(-1)
        else:
            obs_imu = torch.cat(
                (
                    self._robot.data.root_ang_vel_b * self.cfg.normalization.obs_scales.ang_vel,
                    self._base_euler_xyz * self.cfg.normalization.obs_scales.quat,
                ),
                dim=1,
            )

        policy_frame = torch.cat((command_input, obs_imu, obs_motor, self.actions), dim=-1)
        if self.cfg.noise.add_noise:
            policy_frame = policy_frame + (2.0 * torch.rand_like(policy_frame) - 1.0) * self._noise_scale_vec * self.cfg.noise.noise_level

        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1] = policy_frame
        self._critic_history = torch.roll(self._critic_history, shifts=-1, dims=1)
        self._critic_history[:, -1] = privileged_frame

        clip_obs = self.cfg.normalization.clip_observations
        policy_obs = torch.clamp(self._obs_history.reshape(self.num_envs, -1), -clip_obs, clip_obs)
        critic_obs = torch.clamp(self._critic_history.reshape(self.num_envs, -1), -clip_obs, clip_obs)
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name in self._reward_names:
            term = getattr(self, f"_reward_{name}")() * self._reward_scales[name]
            reward += term
            self._episode_sums[name] += term
        if self.cfg.rewards.only_positive_rewards:
            reward = torch.clamp(reward, min=0.0)
        if self._reward_scales["termination"] != 0.0:
            term = self._reward_termination() * self._reward_scales["termination"]
            reward += term
            self._episode_sums["termination"] += term
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_contact = torch.any(
            torch.norm(self._contact_forces[:, self._contact_base_body_ids, :], dim=-1) > 1.0, dim=1
        )
        non_foot_contact = torch.any(
            torch.norm(self._contact_forces[:, self._contact_non_foot_body_ids, :], dim=-1) > 1.0, dim=1
        )
        back_lie = (self._robot.data.projected_gravity_b[:, 2] > 0.0) & non_foot_contact & ~base_contact
        if self._has_trampoline:
            base_xy = self._robot.data.root_pos_w[:, :2]
            trampoline_center_xy = self.scene.env_origins[:, :2]
            out_of_trampoline = torch.linalg.vector_norm(base_xy - trampoline_center_xy, dim=1) > self.cfg.usable_radius
        else:
            out_of_trampoline = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()
        bad_joint_pos = torch.any(~torch.isfinite(joint_pos) | (torch.abs(joint_pos) > 3.5), dim=1)
        bad_joint_vel = torch.any(~torch.isfinite(joint_vel) | (torch.abs(joint_vel) > 60.0), dim=1)
        bad_root_pos = torch.any(~torch.isfinite(self._robot.data.root_pos_w), dim=1)
        bad_root_vel = torch.any(~torch.isfinite(self._robot.data.root_lin_vel_b), dim=1) | torch.any(
            ~torch.isfinite(self._robot.data.root_ang_vel_b), dim=1
        )
        bad_state = bad_joint_pos | bad_joint_vel | bad_root_pos | bad_root_vel

        self._term_base_contact = base_contact
        self._term_back_lie = back_lie
        self._term_out_of_trampoline = out_of_trampoline
        self._term_bad_state = bad_state
        time_out = self.episode_length_buf > self.max_episode_length
        return base_contact | back_lie | out_of_trampoline | bad_state, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = self._all_env_ids
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)
        self._robot.reset(env_ids)
        self._contact_sensor.reset(env_ids)
        if self._has_trampoline and self._trampoline_targets is not None:
            reset_deformable_trampoline(self._trampoline, self._trampoline_targets, env_ids=env_ids)

        if (
            self._manual_command is None
            and self.cfg.commands.curriculum
            and self.common_step_counter > 0
            and self.common_step_counter % self.max_episode_length == 0
        ):
            self._update_command_curriculum(env_ids)

        if self._manual_command is None:
            self._resample_commands(env_ids)
        else:
            self._apply_manual_command(env_ids)
        self._reset_latency_buffers(env_ids)

        self.actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._last_dof_vel[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0
        self._last_contacts[env_ids] = False
        self._obs_history[env_ids] = 0.0
        self._critic_history[env_ids] = 0.0

        joint_pos = self._default_dof_pos[env_ids] + self._sample_uniform(-0.1, 0.1, (len(env_ids), self._action_dim))
        joint_vel = torch.zeros(len(env_ids), self._action_dim, dtype=torch.float, device=self.device)
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        if self._has_trampoline:
            default_root_state[:, 2] += self.cfg.trampoline_surface_height + self.cfg.trampoline_robot_clearance

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self._canonical_joint_ids,
            env_ids=env_ids,
        )
        self._robot.set_joint_position_target(
            self._default_dof_pos[env_ids],
            joint_ids=self._canonical_joint_ids,
            env_ids=env_ids,
        )
        self.episode_length_buf[env_ids] = 0

        extras = {}
        for key in self._episode_sums:
            extras[f"rew_{key}"] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["max_command_x"] = self.cfg.commands.ranges.lin_vel_x[1]
        extras["term_base_contact"] = torch.count_nonzero(self._term_base_contact[env_ids]).item()
        extras["term_back_lie"] = torch.count_nonzero(self._term_back_lie[env_ids]).item()
        extras["term_out_of_trampoline"] = torch.count_nonzero(self._term_out_of_trampoline[env_ids]).item()
        extras["term_bad_state"] = torch.count_nonzero(self._term_bad_state[env_ids]).item()
        extras["term_terminated"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["term_time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["episode"] = extras

    def _get_phase(self) -> torch.Tensor:
        return self.episode_length_buf.to(torch.float) * self.step_dt / self.cfg.rewards.cycle_time

    def _get_gait_phase(self) -> torch.Tensor:
        phase = self._get_phase()
        stance_mask = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        stance_mask[:, 0] = phase < 0.6
        stance_mask[:, 1] = phase > 0.6
        return stance_mask

    def _post_physics_step_callback(self):
        if self._manual_command is None:
            env_ids = (self.episode_length_buf % self._command_resample_interval == 0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        else:
            self._apply_manual_command(self._all_env_ids)
        if self.cfg.domain_rand.push_robots and self._push_interval > 0 and self.common_step_counter % self._push_interval == 0:
            self._push_robots()

    def _resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self._commands[env_ids, 0] = self._sample_uniform(
            self.cfg.commands.ranges.lin_vel_x[0], self.cfg.commands.ranges.lin_vel_x[1], (len(env_ids),)
        )
        self._commands[env_ids, 1] = self._sample_uniform(
            self.cfg.commands.ranges.lin_vel_y[0], self.cfg.commands.ranges.lin_vel_y[1], (len(env_ids),)
        )
        self._commands[env_ids, 2] = self._sample_uniform(
            self.cfg.commands.ranges.ang_vel_yaw[0], self.cfg.commands.ranges.ang_vel_yaw[1], (len(env_ids),)
        )
        self._commands[env_ids, 3] = 0.0
        moving = torch.norm(self._commands[env_ids, :2], dim=1) > self._command_xy_deadzone
        self._commands[env_ids, :2] *= moving.unsqueeze(1)

    def set_manual_command(
        self,
        lin_vel_x: float,
        lin_vel_y: float,
        ang_vel_yaw: float,
        heading: float = 0.0,
    ) -> None:
        self._manual_command = torch.tensor(
            [lin_vel_x, lin_vel_y, ang_vel_yaw, heading],
            device=self.device,
            dtype=torch.float,
        )
        self._apply_manual_command(self._all_env_ids)

    def clear_manual_command(self) -> None:
        self._manual_command = None

    def _apply_manual_command(self, env_ids: torch.Tensor) -> None:
        if self._manual_command is None or len(env_ids) == 0:
            return
        command = self._manual_command.clone()
        if torch.norm(command[:2]) <= self._command_xy_deadzone:
            command[:2] = 0.0
        self._commands[env_ids] = command.unsqueeze(0)

    def _update_command_curriculum(self, env_ids: torch.Tensor):
        if (
            torch.mean(self._episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self._reward_scales["tracking_lin_vel"]
        ):
            min_x, max_x = self.cfg.commands.ranges.lin_vel_x
            self.cfg.commands.ranges.lin_vel_x = (
                max(min_x - 0.5, -self.cfg.commands.max_curriculum),
                min(max_x + 0.5, self.cfg.commands.max_curriculum),
            )

    def _push_robots(self):
        root_velocity = self._robot.data.root_state_w[:, 7:].clone()
        root_velocity[:, :2] = self._sample_uniform(
            -self.cfg.domain_rand.max_push_vel_xy,
            self.cfg.domain_rand.max_push_vel_xy,
            (self.num_envs, 2),
        )
        root_velocity[:, 3:] = self._sample_uniform(
            -self.cfg.domain_rand.max_push_ang_vel,
            self.cfg.domain_rand.max_push_ang_vel,
            (self.num_envs, 3),
        )
        self._robot.write_root_velocity_to_sim(root_velocity)

    def _apply_startup_randomization(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        if self.cfg.domain_rand.randomize_friction:
            env_ids_cpu = env_ids.cpu()
            materials = self._robot.root_physx_view.get_material_properties()
            friction = self._sample_uniform(
                self.cfg.domain_rand.friction_range[0],
                self.cfg.domain_rand.friction_range[1],
                (len(env_ids), 1),
                device="cpu",
            )
            friction_per_shape = friction.expand(-1, materials.shape[1])
            materials[env_ids_cpu, :, 0] = friction_per_shape
            materials[env_ids_cpu, :, 1] = friction_per_shape
            materials[env_ids_cpu, :, 2] = 0.0
            self._robot.root_physx_view.set_material_properties(materials, env_ids_cpu)
            self._env_frictions[env_ids] = friction.to(self.device)

        if self.cfg.domain_rand.randomize_base_mass:
            _invoke_randomization_term(
                randomize_rigid_body_mass,
                self,
                env_ids,
                asset_cfg=SceneEntityCfg("robot", body_ids=self._base_body_ids.tolist()),
                mass_distribution_params=self.cfg.domain_rand.added_base_mass_range,
                operation="add",
            )
        if self.cfg.domain_rand.randomize_link_mass and len(self._non_base_body_ids) > 0:
            _invoke_randomization_term(
                randomize_rigid_body_mass,
                self,
                env_ids,
                asset_cfg=SceneEntityCfg("robot", body_ids=self._non_base_body_ids.tolist()),
                mass_distribution_params=self.cfg.domain_rand.multiplied_link_mass_range,
                operation="scale",
            )
        if self.cfg.domain_rand.randomize_base_com:
            randomize_rigid_body_com(
                self,
                env_ids,
                {
                    "x": self.cfg.domain_rand.added_base_com_range,
                    "y": self.cfg.domain_rand.added_base_com_range,
                    "z": self.cfg.domain_rand.added_base_com_range,
                },
                SceneEntityCfg("robot", body_ids=self._base_body_ids.tolist()),
            )
        if self.cfg.domain_rand.randomize_pd_gains:
            _invoke_randomization_term(
                randomize_actuator_gains,
                self,
                env_ids,
                asset_cfg=SceneEntityCfg("robot"),
                stiffness_distribution_params=self.cfg.domain_rand.stiffness_multiplier_range,
                damping_distribution_params=self.cfg.domain_rand.damping_multiplier_range,
                operation="scale",
            )
        if self.cfg.domain_rand.randomize_motor_zero_offset:
            self._motor_zero_offsets[env_ids] = self._sample_uniform(
                self.cfg.domain_rand.motor_zero_offset_range[0],
                self.cfg.domain_rand.motor_zero_offset_range[1],
                (len(env_ids), self._action_dim),
            )

    def _reset_latency_buffers(self, env_ids: torch.Tensor):
        if self.cfg.domain_rand.add_cmd_action_latency:
            self._cmd_action_latency_buffer[env_ids] = 0.0
            if self.cfg.domain_rand.randomize_cmd_action_latency:
                self._cmd_action_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_cmd_action_latency[0],
                    self.cfg.domain_rand.range_cmd_action_latency[1] + 1,
                    (len(env_ids),),
                    device=self.device,
                )
            else:
                self._cmd_action_latency_simstep[env_ids] = self.cfg.domain_rand.range_cmd_action_latency[1]

        if self.cfg.domain_rand.add_obs_latency:
            self._obs_motor_latency_buffer[env_ids] = 0.0
            self._obs_imu_latency_buffer[env_ids] = 0.0
            if self.cfg.domain_rand.randomize_obs_motor_latency:
                self._obs_motor_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_obs_motor_latency[0],
                    self.cfg.domain_rand.range_obs_motor_latency[1] + 1,
                    (len(env_ids),),
                    device=self.device,
                )
            else:
                self._obs_motor_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_motor_latency[1]
            if self.cfg.domain_rand.randomize_obs_imu_latency:
                self._obs_imu_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_obs_imu_latency[0],
                    self.cfg.domain_rand.range_obs_imu_latency[1] + 1,
                    (len(env_ids),),
                    device=self.device,
                )
            else:
                self._obs_imu_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_imu_latency[1]

    def _update_obs_latency_buffers(self):
        if self.cfg.domain_rand.add_obs_latency and self.cfg.domain_rand.randomize_obs_motor_latency:
            joint_pos = self._joint_pos()
            joint_vel = self._joint_vel()
            q = (joint_pos - self._default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
            dq = joint_vel * self.cfg.normalization.obs_scales.dof_vel
            self._obs_motor_latency_buffer[:, :, 1:] = self._obs_motor_latency_buffer[:, :, :-1]
            self._obs_motor_latency_buffer[:, :, 0] = torch.cat((q, dq), dim=1)
        if self.cfg.domain_rand.add_obs_latency and self.cfg.domain_rand.randomize_obs_imu_latency:
            roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_link_state_w[:, 3:7])
            base_euler = self._wrap_euler(torch.stack((roll, pitch, yaw), dim=1))
            self._obs_imu_latency_buffer[:, :, 1:] = self._obs_imu_latency_buffer[:, :, :-1]
            self._obs_imu_latency_buffer[:, :, 0] = torch.cat(
                (
                    self._robot.data.root_ang_vel_b * self.cfg.normalization.obs_scales.ang_vel,
                    base_euler * self.cfg.normalization.obs_scales.quat,
                ),
                dim=1,
            )

    def _update_state_buffers(self):
        self._contact_forces = self._contact_sensor.data.net_forces_w
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_link_state_w[:, 3:7])
        self._base_euler_xyz = self._wrap_euler(torch.stack((roll, pitch, yaw), dim=1))
        if self._has_trampoline and self._trampoline_center_node_ids is not None:
            self._measured_heights = trampoline_center_heights(self._trampoline, self._trampoline_center_node_ids)
            update_trampoline_visual_height(self.scene.env_origins, self._trampoline_visual_translate_ops, self._measured_heights)
        else:
            self._measured_heights.zero_()

    def _build_noise_scale_vec(self) -> torch.Tensor:
        noise = torch.zeros(self._policy_frame_dim, dtype=torch.float, device=self.device)
        scales = self.cfg.noise.noise_scales
        obs_scales = self.cfg.normalization.obs_scales
        noise[5:8] = scales.ang_vel * obs_scales.ang_vel
        noise[8:11] = scales.quat
        noise[11:23] = scales.dof_pos * obs_scales.dof_pos
        noise[23:35] = scales.dof_vel * obs_scales.dof_vel
        return noise

    def _build_policy_obs_names(self) -> list[str]:
        names = [
            "sin_phase",
            "cos_phase",
            "cmd_lin_vel_x",
            "cmd_lin_vel_y",
            "cmd_ang_vel_yaw",
            "imu_ang_vel_x",
            "imu_ang_vel_y",
            "imu_ang_vel_z",
            "imu_roll",
            "imu_pitch",
            "imu_yaw",
        ]
        names += [f"joint_pos_rel_{joint_name}" for joint_name in GO2_CSV_JOINT_NAMES]
        names += [f"joint_vel_{joint_name}" for joint_name in GO2_CSV_JOINT_NAMES]
        names += [f"last_action_{joint_name}" for joint_name in GO2_CSV_JOINT_NAMES]
        return names

    def _build_critic_obs_names(self) -> list[str]:
        names = [
            "sin_phase",
            "cos_phase",
            "cmd_lin_vel_x",
            "cmd_lin_vel_y",
            "cmd_ang_vel_yaw",
        ]
        names += [f"dof_pos_minus_pd_target_{joint_name}" for joint_name in GO2_CSV_JOINT_NAMES]
        names += [f"dof_pos_{joint_name}" for joint_name in GO2_CSV_JOINT_NAMES]
        names += [f"dof_vel_{joint_name}" for joint_name in GO2_CSV_JOINT_NAMES]
        names += [f"last_action_{joint_name}" for joint_name in GO2_CSV_JOINT_NAMES]
        names += ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
        names += ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
        names += ["base_roll", "base_pitch", "base_yaw"]
        names += ["env_friction", "body_mass_div10", "stance_mask_left", "stance_mask_right"]
        names += ["foot_contact_fl", "foot_contact_fr", "foot_contact_rl", "foot_contact_rr"]
        return names

    def get_export_metadata(self, run_path: str) -> dict[str, object]:
        metadata = {
            "run_path": run_path,
            "joint_names": list(self._canonical_joint_names),
            "joint_stiffness": self._robot.data.joint_stiffness[0, self._canonical_joint_ids].cpu().tolist(),
            "joint_damping": self._robot.data.joint_damping[0, self._canonical_joint_ids].cpu().tolist(),
            "default_joint_pos": self._default_dof_pos[0].cpu().tolist(),
            "command_names": ["lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"],
            "observation_names": self._policy_obs_names,
            "observation_history_lengths": [self._policy_frame_stack] * len(self._policy_obs_names),
            "critic_observation_names": self._critic_obs_names,
            "critic_observation_history_lengths": [self._critic_frame_stack] * len(self._critic_obs_names),
            "action_scale": self._action_scale.squeeze(0).cpu().tolist(),
            "anchor_body_name": self.cfg.anchor_body_name,
            "body_names": list(self._robot.data.body_names),
            "symmetry_observation_frame_permutation": list(GO2_JUMP_OBS_PERMUTATION),
            "symmetry_action_permutation": list(GO2_JUMP_ACTION_PERMUTATION),
        }
        if self._has_trampoline:
            metadata.update({
                "trampoline_radius": self.cfg.trampoline_radius,
                "trampoline_pin_width": self.cfg.trampoline_pin_width,
                "trampoline_surface_height": self.cfg.trampoline_surface_height,
                "usable_radius": self.cfg.usable_radius,
            })
        return metadata

    def _reward_jump(self) -> torch.Tensor:
        contact = self._contact_forces[:, self._contact_foot_body_ids, 2] > 5.0
        stance_mask = self._get_gait_phase()
        jump = (
            (contact[:, 0] == contact[:, 1])
            & (contact[:, 1] == contact[:, 2])
            & (contact[:, 2] == contact[:, 3])
            & (contact[:, 3] == stance_mask[:, 0].bool())
        )
        return jump.float() * self._command_xy_is_moving()

    def _reward_default_hip_pos(self) -> torch.Tensor:
        joint_pos = self._joint_pos()
        joint_diff = (
            torch.abs(joint_pos[:, 0])
            + torch.abs(joint_pos[:, 3])
            + torch.abs(joint_pos[:, 6])
            + torch.abs(joint_pos[:, 9])
        )
        return torch.exp(-joint_diff * 4.0)

    def _reward_feet_clearance(self) -> torch.Tensor:
        feet_height = self._robot.data.body_pos_w[:, self._foot_body_ids, 2] - self._measured_heights - 0.02
        swing_mask = 1.0 - self._get_gait_phase()
        reward = torch.clamp(feet_height, min=0.0, max=0.05)
        reward = torch.sum(reward * swing_mask[:, :1].repeat(1, 4), dim=1)
        return reward * self._command_xy_is_moving()

    def _reward_lin_vel_z(self) -> torch.Tensor:
        return torch.exp(-torch.abs(self._robot.data.root_lin_vel_b[:, 2]))

    def _reward_ang_vel_xy(self) -> torch.Tensor:
        return torch.exp(-torch.norm(torch.abs(self._robot.data.root_ang_vel_b[:, :2]), dim=1))

    def _reward_orientation(self) -> torch.Tensor:
        return torch.exp(-torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) * 10.0)

    def _reward_base_height(self) -> torch.Tensor:
        base_height = torch.mean(self._robot.data.root_link_state_w[:, 2:3] - self._measured_heights, dim=1)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 10.0) * (
            ~self._command_xy_is_moving()
        )

    def _reward_torques(self) -> torch.Tensor:
        return torch.sum(torch.abs(self._applied_torque()), dim=1)

    def _reward_dof_vel(self) -> torch.Tensor:
        return torch.sum(torch.square(self._joint_vel()), dim=1)

    def _reward_dof_acc(self) -> torch.Tensor:
        dof_vel_delta = torch.clamp(self._last_dof_vel - self._joint_vel(), min=-20.0, max=20.0)
        return torch.sum(torch.square(dof_vel_delta), dim=1).clamp(max=1000.0)

    def _reward_action_rate(self) -> torch.Tensor:
        return torch.sum(torch.square(self._last_actions - self.actions), dim=1)

    def _reward_collision(self) -> torch.Tensor:
        return torch.sum(
            (torch.norm(self._contact_forces[:, self._contact_penalized_body_ids, :], dim=-1) > 0.1).float(),
            dim=1,
        )

    def _reward_termination(self) -> torch.Tensor:
        return (self.reset_buf & ~self.reset_time_outs).float()

    def _reward_tracking_lin_vel(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) * (
            self._command_xy_is_moving()
        )
        reward += torch.exp(-torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1) / self.cfg.rewards.tracking_sigma) * (
            ~self._command_xy_is_moving()
        )
        return reward

    def _reward_tracking_ang_vel(self) -> torch.Tensor:
        ang_vel_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        reward = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma) * (torch.abs(self._commands[:, 2]) > 0.2)
        reward += torch.exp(-torch.abs(self._robot.data.root_ang_vel_b[:, 2]) / self.cfg.rewards.tracking_sigma) * (
            torch.abs(self._commands[:, 2]) < 0.2
        )
        return reward

    def _reward_feet_air_time(self) -> torch.Tensor:
        contact = self._contact_forces[:, self._contact_foot_body_ids, 2] > 1.0
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        first_contact = (self._feet_air_time > 0.0) * contact_filt
        self._feet_air_time += self.step_dt
        reward = torch.sum((self._feet_air_time - 0.5) * first_contact.float(), dim=1)
        reward *= self._command_xy_is_moving()
        self._feet_air_time *= (~contact_filt).float()
        return reward

    def _reward_feet_stumble(self) -> torch.Tensor:
        return torch.any(
            torch.norm(self._contact_forces[:, self._contact_foot_body_ids, :2], dim=2)
            > 5.0 * torch.abs(self._contact_forces[:, self._contact_foot_body_ids, 2]),
            dim=1,
        ).float()

    def _reward_default_pos(self) -> torch.Tensor:
        joint_error = torch.abs(self._joint_pos() - self._default_dof_pos)
        return torch.sum(torch.clamp(joint_error, max=1.0), dim=1)

    def _reward_feet_contact_forces(self) -> torch.Tensor:
        return torch.sum(
            (
                torch.norm(self._contact_forces[:, self._contact_foot_body_ids, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _joint_pos(self) -> torch.Tensor:
        return self._robot.data.joint_pos[:, self._canonical_joint_ids]

    def _joint_vel(self) -> torch.Tensor:
        return self._robot.data.joint_vel[:, self._canonical_joint_ids]

    def _applied_torque(self) -> torch.Tensor:
        return self._robot.data.applied_torque[:, self._canonical_joint_ids]

    @staticmethod
    def _wrap_euler(euler_xyz: torch.Tensor) -> torch.Tensor:
        wrapped = euler_xyz.clone()
        wrapped[wrapped > math.pi] -= 2.0 * math.pi
        return wrapped

    def _sample_uniform(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        sample_device = self.device if device is None else device
        return low + (high - low) * torch.rand(shape, device=sample_device)

    def _command_xy_is_moving(self) -> torch.Tensor:
        return torch.norm(self._commands[:, :2], dim=1) > self._command_xy_deadzone

    def _to_tensor(self, ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor(list(ids), dtype=torch.long, device=self.device)
