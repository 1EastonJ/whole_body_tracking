from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from whole_body_tracking.utils.trampoline_deformable import (
    TRAMPOLINE_DR_DAMPING_SCALE_RANGE,
    TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE,
    TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE,
    TRAMPOLINE_DR_MASS_RANGE,
    TRAMPOLINE_DR_POISSONS_RATIO_RANGE,
    TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    build_trampoline_kinematic_targets,
    reset_deformable_trampoline,
    set_trampoline_material_properties,
    trampoline_mesh_prim_path,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material_shared(ManagerTermBase):
    """Randomize rigid-body materials while keeping one shared sample per environment.

    This mirrors mjlab's shared foot-friction randomization more closely than IsaacLab's
    default material randomizer, which samples each selected shape independently.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params['asset_cfg']
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material_shared' not supported for asset: "
                f"'{self.asset_cfg.name}' with type: '{type(self.asset)}'."
            )

        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
                self.num_shapes_per_body.append(link_physx_view.max_shapes)

            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material_shared' failed to parse the number of "
                    f"shapes per body. Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            self.num_shapes_per_body = None

        static_friction_range = cfg.params.get('static_friction_range', (1.0, 1.0))
        dynamic_friction_range = cfg.params.get('dynamic_friction_range', (1.0, 1.0))
        restitution_range = cfg.params.get('restitution_range', (0.0, 0.0))
        num_buckets = int(cfg.params.get('num_buckets', 1))

        ranges = torch.tensor(
            [static_friction_range, dynamic_friction_range, restitution_range],
            device='cpu',
        )
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device='cpu')

        if cfg.params.get('make_consistent', False):
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        del env, static_friction_range, dynamic_friction_range, restitution_range, make_consistent

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device='cpu')
        else:
            env_ids = env_ids.cpu()

        materials = self.asset.root_physx_view.get_material_properties()
        bucket_ids = torch.randint(0, int(num_buckets), (len(env_ids),), device='cpu')
        sampled_materials = self.material_buckets[bucket_ids]

        if self.num_shapes_per_body is None or asset_cfg.body_ids == slice(None):
            total_num_shapes = self.asset.root_physx_view.max_shapes
            materials[env_ids] = sampled_materials[:, None, :].expand(-1, total_num_shapes, -1)
        else:
            for body_id in asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = sampled_materials[:, None, :]

        self.asset.root_physx_view.set_material_properties(materials, env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal['add', 'scale', 'abs'] = 'abs',
    distribution: Literal['uniform', 'log_uniform', 'gaussian'] = 'uniform',
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term('joint_pos')._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device='cpu')
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device='cpu')
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device='cpu')

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ['x', 'y', 'z']]
    ranges = torch.tensor(range_list, device='cpu')
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device='cpu').unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def _resolve_deformable_mass_attrs(trampoline: DeformableObject) -> list:
    stage = get_current_stage()
    attrs = []
    for prim_path in sim_utils.find_matching_prim_paths(trampoline_mesh_prim_path(trampoline.cfg.prim_path)):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid deformable trampoline mesh prim: '{prim_path}'.")
        attr = prim.GetAttribute("physics:mass")
        if not attr.IsValid():
            attr = prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float)
        attrs.append(attr)
    return attrs


def _get_cached_deformable_mass_attrs(env: ManagerBasedEnv, trampoline: DeformableObject, asset_name: str) -> list:
    cache = getattr(env, "_deformable_mass_attrs_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_deformable_mass_attrs_cache", cache)

    attrs = cache.get(asset_name)
    if attrs is None:
        attrs = _resolve_deformable_mass_attrs(trampoline)
        cache[asset_name] = attrs
    return attrs


def randomize_deformable_trampoline_material_event(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    youngs_modulus_range: tuple[float, float] = TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    mass_range: tuple[float, float] = TRAMPOLINE_DR_MASS_RANGE,
    dynamic_friction_range: tuple[float, float] = TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE,
    elasticity_damping_range: tuple[float, float] = TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE,
    damping_scale_range: tuple[float, float] = TRAMPOLINE_DR_DAMPING_SCALE_RANGE,
    poissons_ratio_range: tuple[float, float] = TRAMPOLINE_DR_POISSONS_RATIO_RANGE,
):
    """Randomize deformable trampoline material and mass for selected environments."""
    trampoline: DeformableObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=trampoline.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=trampoline.device, dtype=torch.long).reshape(-1)

    if trampoline.material_physx_view is None:
        raise RuntimeError("Failed to create deformable trampoline material view.")

    sample_shape = (len(env_ids),)
    youngs_moduli = math_utils.sample_uniform(*youngs_modulus_range, sample_shape, device=trampoline.device).to(
        dtype=torch.float32
    )
    dynamic_frictions = math_utils.sample_uniform(*dynamic_friction_range, sample_shape, device=trampoline.device).to(
        dtype=torch.float32
    )
    elasticity_dampings = math_utils.sample_uniform(*elasticity_damping_range, sample_shape, device=trampoline.device).to(
        dtype=torch.float32
    )
    damping_scales = math_utils.sample_uniform(*damping_scale_range, sample_shape, device=trampoline.device).to(
        dtype=torch.float32
    )
    poissons_ratios = math_utils.sample_uniform(*poissons_ratio_range, sample_shape, device=trampoline.device).to(
        dtype=torch.float32
    )
    masses = math_utils.sample_uniform(*mass_range, sample_shape, device=trampoline.device).to(dtype=torch.float32)

    set_trampoline_material_properties(
        trampoline.material_physx_view,
        env_ids,
        youngs_moduli=youngs_moduli,
        dynamic_frictions=dynamic_frictions,
        elasticity_dampings=elasticity_dampings,
        damping_scales=damping_scales,
        poissons_ratios=poissons_ratios,
    )

    mass_attrs = _get_cached_deformable_mass_attrs(env, trampoline, asset_cfg.name)
    with Sdf.ChangeBlock():
        for env_id, mass in zip(env_ids.tolist(), masses.tolist(), strict=True):
            mass_attrs[env_id].Set(float(mass))


def reset_deformable_trampoline_event(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pin_width: float | None = None,
    pin_radius: float | torch.Tensor | None = None,
    randomize_material: bool = False,
    youngs_modulus_range: tuple[float, float] = TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE,
    mass_range: tuple[float, float] = TRAMPOLINE_DR_MASS_RANGE,
    dynamic_friction_range: tuple[float, float] = TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE,
    elasticity_damping_range: tuple[float, float] = TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE,
    damping_scale_range: tuple[float, float] = TRAMPOLINE_DR_DAMPING_SCALE_RANGE,
    poissons_ratio_range: tuple[float, float] = TRAMPOLINE_DR_POISSONS_RATIO_RANGE,
):
    """Reset a deformable trampoline back to its default nodal state on episode reset."""
    trampoline: DeformableObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=trampoline.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=trampoline.device, dtype=torch.long).reshape(-1)

    if randomize_material:
        randomize_deformable_trampoline_material_event(
            env,
            env_ids,
            asset_cfg,
            youngs_modulus_range=youngs_modulus_range,
            mass_range=mass_range,
            dynamic_friction_range=dynamic_friction_range,
            elasticity_damping_range=elasticity_damping_range,
            damping_scale_range=damping_scale_range,
            poissons_ratio_range=poissons_ratio_range,
        )

    cache = getattr(env, "_deformable_reset_targets_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_deformable_reset_targets_cache", cache)

    trampoline_targets = cache.get(asset_cfg.name)
    if trampoline_targets is None or trampoline_targets.shape != trampoline.data.nodal_kinematic_target.shape:
        trampoline_targets, _, _ = build_trampoline_kinematic_targets(
            trampoline.data.default_nodal_state_w,
            trampoline.data.nodal_kinematic_target,
            pin_width=pin_width,
            pin_radius=pin_radius,
        )
        cache[asset_cfg.name] = trampoline_targets

    reset_deformable_trampoline(trampoline, trampoline_targets, env_ids=env_ids)
