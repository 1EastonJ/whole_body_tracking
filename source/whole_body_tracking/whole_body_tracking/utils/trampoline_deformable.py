from __future__ import annotations

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg

TRAMPOLINE_RADIUS = 1.5
TRAMPOLINE_THICKNESS = 0.05
TRAMPOLINE_TOP_Z = 0.0
TRAMPOLINE_CENTER_Z = TRAMPOLINE_TOP_Z - 0.5 * TRAMPOLINE_THICKNESS
TRAMPOLINE_PIN_RADIUS = TRAMPOLINE_RADIUS
TRAMPOLINE_PIN_WIDTH = 0.0
TRAMPOLINE_MASS = 7.5
TRAMPOLINE_YOUNGS_MODULUS = 1.5e5
TRAMPOLINE_DYNAMIC_FRICTION = 0.8
TRAMPOLINE_POISSONS_RATIO = 0.35
TRAMPOLINE_ELASTICITY_DAMPING = 0.005
TRAMPOLINE_DAMPING_SCALE = 0.5
TRAMPOLINE_SIM_RESOLUTION = 15
MIXED_RESET_STATIC_HEIGHT_OFFSET = 0.0
TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE = (1.0e5, 3.0e5)
TRAMPOLINE_DR_MASS_RANGE = (7.5, 20.0)
MIXED_RESET_DROP_HEIGHT_RANGE = (-0.01, 0.01)
TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE = (0.4, 1.2)
TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE = (0.003, 0.008)
TRAMPOLINE_DR_DAMPING_SCALE_RANGE = (0.4, 0.6)
TRAMPOLINE_DR_POISSONS_RATIO_RANGE = (0.25, 0.45)


def make_trampoline_cfg(
    prim_path: str,
    *,
    center_z: float = TRAMPOLINE_CENTER_Z,
    mass: float = TRAMPOLINE_MASS,
    youngs_modulus: float = TRAMPOLINE_YOUNGS_MODULUS,
    sim_resolution: int = TRAMPOLINE_SIM_RESOLUTION,
    debug_vis: bool = False,
) -> DeformableObjectCfg:
    """Create the shared deformable trampoline configuration."""
    return DeformableObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCylinderCfg(
            radius=TRAMPOLINE_RADIUS,
            height=TRAMPOLINE_THICKNESS,
            axis="Z",
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                solver_position_iteration_count=24,
                vertex_velocity_damping=0.05,
                sleep_damping=1.0,
                sleep_threshold=0.01,
                settling_threshold=0.02,
                self_collision=False,
                simulation_hexahedral_resolution=sim_resolution,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), metallic=0.05),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                dynamic_friction=TRAMPOLINE_DYNAMIC_FRICTION,
                youngs_modulus=youngs_modulus,
                poissons_ratio=TRAMPOLINE_POISSONS_RATIO,
                elasticity_damping=TRAMPOLINE_ELASTICITY_DAMPING,
                damping_scale=TRAMPOLINE_DAMPING_SCALE,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, center_z)),
        debug_vis=debug_vis,
    )


def build_trampoline_kinematic_targets(
    default_nodal_state_w: torch.Tensor,
    nodal_kinematic_target: torch.Tensor,
    pin_width: float | None = None,
    pin_radius: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create kinematic targets that pin the outer rim of a deformable trampoline."""
    targets = nodal_kinematic_target.clone()
    targets[..., :3] = default_nodal_state_w[..., :3]
    targets[..., 3] = 1.0

    nodal_pos = default_nodal_state_w[..., :3]
    xy_min = nodal_pos[..., :2].amin(dim=1, keepdim=True)
    xy_max = nodal_pos[..., :2].amax(dim=1, keepdim=True)
    center_xy = 0.5 * (xy_min + xy_max)
    radial_distance = torch.linalg.vector_norm(nodal_pos[..., :2] - center_xy, dim=-1)

    edge_radius = radial_distance.max(dim=1, keepdim=True).values
    if pin_radius is None:
        if pin_width is None:
            pin_radius = TRAMPOLINE_PIN_RADIUS
        else:
            pin_threshold = torch.clamp(edge_radius - pin_width, min=0.0)
            pinned_mask = radial_distance >= pin_threshold
            min_radial_distance = radial_distance.amin(dim=1, keepdim=True)
            center_candidates = radial_distance <= min_radial_distance + 1.0e-6
            center_candidate_z = torch.where(
                center_candidates,
                nodal_pos[..., 2],
                torch.full_like(nodal_pos[..., 2], -torch.inf),
            )
            center_node_ids = center_candidate_z.argmax(dim=1)

            targets[..., 3] = torch.where(
                pinned_mask,
                torch.zeros_like(targets[..., 3]),
                torch.ones_like(targets[..., 3]),
            )
            return targets, pinned_mask, center_node_ids

    pin_radius_column = _as_column_tensor(pin_radius, device=radial_distance.device).to(dtype=radial_distance.dtype)
    pin_threshold = torch.minimum(torch.clamp(pin_radius_column, min=0.0), edge_radius)
    pinned_mask = radial_distance >= pin_threshold
    min_radial_distance = radial_distance.amin(dim=1, keepdim=True)
    center_candidates = radial_distance <= min_radial_distance + 1.0e-6
    center_candidate_z = torch.where(
        center_candidates,
        nodal_pos[..., 2],
        torch.full_like(nodal_pos[..., 2], -torch.inf),
    )
    center_node_ids = center_candidate_z.argmax(dim=1)

    targets[..., 3] = torch.where(
        pinned_mask,
        torch.zeros_like(targets[..., 3]),
        torch.ones_like(targets[..., 3]),
    )
    return targets, pinned_mask, center_node_ids


def reset_deformable_trampoline(
    trampoline: DeformableObject,
    trampoline_targets: torch.Tensor,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Reset selected trampoline instances back to their default nodal state."""
    if env_ids is None:
        env_ids = torch.arange(
            trampoline.data.default_nodal_state_w.shape[0], device=trampoline.device, dtype=torch.long
        )
    else:
        env_ids = torch.as_tensor(env_ids, device=trampoline.device, dtype=torch.long).reshape(-1)

    default_nodal_state = trampoline.data.default_nodal_state_w.index_select(0, env_ids)
    target_state = trampoline_targets.index_select(0, env_ids)
    trampoline.write_nodal_state_to_sim(default_nodal_state, env_ids=env_ids)
    trampoline.write_nodal_kinematic_target_to_sim(target_state, env_ids=env_ids)
    trampoline.reset(env_ids=env_ids)


def trampoline_center_heights(trampoline: DeformableObject, center_node_ids: torch.Tensor) -> torch.Tensor:
    """Return the world-frame height of the trampoline center node in each env."""
    env_ids = torch.arange(center_node_ids.shape[0], device=trampoline.device, dtype=torch.long)
    center_node_ids = center_node_ids.to(device=trampoline.device, dtype=torch.long)
    return trampoline.data.nodal_pos_w[env_ids, center_node_ids, 2].unsqueeze(-1)


def build_trampoline_visual_translate_ops(num_envs: int) -> list[UsdGeom.XformOp] | None:
    """Resolve translate ops for optional plain-cylinder trampoline visuals."""
    stage = get_current_stage()
    ops: list[UsdGeom.XformOp] = []
    for env_id in range(num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/TrampolineVisual")
        if not prim.IsValid():
            return None
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp(opSuffix="trampoline")
        ops.append(translate_op)
    return ops


def update_trampoline_visual_height(
    env_origins: torch.Tensor,
    visual_translate_ops: list[UsdGeom.XformOp] | None,
    measured_heights: torch.Tensor,
) -> None:
    """Update the optional plain-cylinder trampoline visuals to track the center height."""
    if visual_translate_ops is None:
        return

    del env_origins
    heights = measured_heights.detach().view(-1).cpu().tolist()
    for op, height in zip(visual_translate_ops, heights, strict=True):
        current = op.Get()
        x = 0.0 if current is None else float(current[0])
        y = 0.0 if current is None else float(current[1])
        op.Set(Gf.Vec3d(x, y, float(height)))


def trampoline_mesh_prim_path(root_prim_path: str) -> str:
    """Return the mesh prim path inside a spawned mesh-cylinder trampoline."""
    return f"{root_prim_path}/geometry/mesh"


def _get_trampoline_material_property(material_view, plural_getter_name: str, singular_getter_name: str) -> torch.Tensor:
    """Read a deformable material property from whichever material-view API is available."""
    getter = getattr(material_view, plural_getter_name, None)
    if getter is not None:
        values = getter()
    else:
        values = getattr(material_view, singular_getter_name)()
    return torch.as_tensor(values, device="cpu", dtype=torch.float32).clone()


def get_trampoline_youngs_moduli(material_view) -> torch.Tensor:
    """Read Young's modulus values from the available material view API."""
    return _get_trampoline_material_property(material_view, "get_youngs_moduli", "get_youngs_modulus")


def get_trampoline_dynamic_frictions(material_view) -> torch.Tensor:
    """Read deformable material dynamic friction values."""
    return _get_trampoline_material_property(material_view, "get_dynamic_frictions", "get_dynamic_friction")


def get_trampoline_elasticity_dampings(material_view) -> torch.Tensor:
    """Read deformable material elasticity damping values."""
    return _get_trampoline_material_property(material_view, "get_elasticity_dampings", "get_damping")


def get_trampoline_damping_scales(material_view) -> torch.Tensor:
    """Read deformable material damping scale values."""
    return _get_trampoline_material_property(material_view, "get_damping_scales", "get_damping_scale")


def get_trampoline_poissons_ratios(material_view) -> torch.Tensor:
    """Read deformable material Poisson's ratio values."""
    return _get_trampoline_material_property(material_view, "get_poissons_ratios", "get_poissons_ratio")


def _as_column_tensor(values: torch.Tensor, *, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert material properties to the column layout expected by the PhysX tensor API."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"Expected a scalar, vector, or column tensor, got shape {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _set_trampoline_material_property(
    material_view,
    values: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    plural_setter_name: str,
    singular_setter_name: str,
    singular_getter_name: str,
    property_name: str,
) -> None:
    """Write a deformable material property using whichever material-view API is available."""
    env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1).contiguous()

    setter = getattr(material_view, plural_setter_name, None)
    if setter is not None:
        values = _as_column_tensor(values)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} {property_name} values, got {values.shape[0]}.")
        setter(values, indices=env_ids)
    else:
        # The low-level PhysX tensor view expects a full `(count, 1)` material buffer
        # even when `indices` selects only a subset of environments.
        current_values = _as_column_tensor(getattr(material_view, singular_getter_name)()).clone()
        env_ids = env_ids.to(device=current_values.device)
        values = _as_column_tensor(values, device=current_values.device)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} {property_name} values, got {values.shape[0]}.")
        current_values[env_ids] = values
        getattr(material_view, singular_setter_name)(current_values, indices=env_ids)


def set_trampoline_youngs_moduli(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write Young's modulus values using whichever material-view API is available."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_youngs_moduli",
        singular_setter_name="set_youngs_modulus",
        singular_getter_name="get_youngs_modulus",
        property_name="Young's modulus",
    )


def set_trampoline_dynamic_frictions(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material dynamic friction values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_dynamic_frictions",
        singular_setter_name="set_dynamic_friction",
        singular_getter_name="get_dynamic_friction",
        property_name="dynamic friction",
    )


def set_trampoline_elasticity_dampings(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material elasticity damping values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_elasticity_dampings",
        singular_setter_name="set_damping",
        singular_getter_name="get_damping",
        property_name="elasticity damping",
    )


def set_trampoline_damping_scales(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material damping scale values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_damping_scales",
        singular_setter_name="set_damping_scale",
        singular_getter_name="get_damping_scale",
        property_name="damping scale",
    )


def set_trampoline_poissons_ratios(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write deformable material Poisson's ratio values."""
    _set_trampoline_material_property(
        material_view,
        values,
        env_ids,
        plural_setter_name="set_poissons_ratios",
        singular_setter_name="set_poissons_ratio",
        singular_getter_name="get_poissons_ratio",
        property_name="Poisson's ratio",
    )


def set_trampoline_material_properties(
    material_view,
    env_ids: torch.Tensor,
    *,
    youngs_moduli: torch.Tensor | None = None,
    dynamic_frictions: torch.Tensor | None = None,
    elasticity_dampings: torch.Tensor | None = None,
    damping_scales: torch.Tensor | None = None,
    poissons_ratios: torch.Tensor | None = None,
) -> None:
    """Write randomized deformable trampoline material properties for selected environments."""
    if youngs_moduli is not None:
        set_trampoline_youngs_moduli(material_view, youngs_moduli, env_ids)
    if dynamic_frictions is not None:
        set_trampoline_dynamic_frictions(material_view, dynamic_frictions, env_ids)
    if elasticity_dampings is not None:
        set_trampoline_elasticity_dampings(material_view, elasticity_dampings, env_ids)
    if damping_scales is not None:
        set_trampoline_damping_scales(material_view, damping_scales, env_ids)
    if poissons_ratios is not None:
        set_trampoline_poissons_ratios(material_view, poissons_ratios, env_ids)


__all__ = [
    "MIXED_RESET_DROP_HEIGHT_RANGE",
    "MIXED_RESET_STATIC_HEIGHT_OFFSET",
    "TRAMPOLINE_CENTER_Z",
    "TRAMPOLINE_DAMPING_SCALE",
    "TRAMPOLINE_DR_DAMPING_SCALE_RANGE",
    "TRAMPOLINE_DR_DYNAMIC_FRICTION_RANGE",
    "TRAMPOLINE_DR_ELASTICITY_DAMPING_RANGE",
    "TRAMPOLINE_DR_MASS_RANGE",
    "TRAMPOLINE_DR_POISSONS_RATIO_RANGE",
    "TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE",
    "TRAMPOLINE_DYNAMIC_FRICTION",
    "TRAMPOLINE_ELASTICITY_DAMPING",
    "TRAMPOLINE_MASS",
    "TRAMPOLINE_PIN_RADIUS",
    "TRAMPOLINE_PIN_WIDTH",
    "TRAMPOLINE_POISSONS_RATIO",
    "TRAMPOLINE_RADIUS",
    "TRAMPOLINE_SIM_RESOLUTION",
    "TRAMPOLINE_THICKNESS",
    "TRAMPOLINE_TOP_Z",
    "TRAMPOLINE_YOUNGS_MODULUS",
    "build_trampoline_kinematic_targets",
    "build_trampoline_visual_translate_ops",
    "get_trampoline_damping_scales",
    "get_trampoline_dynamic_frictions",
    "get_trampoline_elasticity_dampings",
    "get_trampoline_poissons_ratios",
    "get_trampoline_youngs_moduli",
    "make_trampoline_cfg",
    "reset_deformable_trampoline",
    "set_trampoline_damping_scales",
    "set_trampoline_dynamic_frictions",
    "set_trampoline_elasticity_dampings",
    "set_trampoline_material_properties",
    "set_trampoline_poissons_ratios",
    "set_trampoline_youngs_moduli",
    "trampoline_center_heights",
    "trampoline_mesh_prim_path",
    "update_trampoline_visual_height",
]
