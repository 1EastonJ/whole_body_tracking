from __future__ import annotations

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg

TRAMPOLINE_RADIUS = 5.0
TRAMPOLINE_THICKNESS = 0.5
TRAMPOLINE_TOP_Z = 0.0
TRAMPOLINE_CENTER_Z = TRAMPOLINE_TOP_Z - 0.5 * TRAMPOLINE_THICKNESS
TRAMPOLINE_PIN_WIDTH = 3.0
TRAMPOLINE_MASS = 10.0
TRAMPOLINE_YOUNGS_MODULUS = 3.0e7
TRAMPOLINE_SIM_RESOLUTION = 20
TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE = (3.0e7, 3.0e7)
TRAMPOLINE_DR_MASS_RANGE = (10.0, 10.0)


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
                solver_position_iteration_count=48,
                vertex_velocity_damping=0.05,
                sleep_damping=1.0,
                sleep_threshold=0.01,
                settling_threshold=0.02,
                self_collision=False,
                simulation_hexahedral_resolution=sim_resolution,
                contact_offset=0.05,
                rest_offset=0.02,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95), metallic=0.05),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                dynamic_friction=0.8,
                youngs_modulus=youngs_modulus,
                poissons_ratio=0.35,
                elasticity_damping=0.02,
                damping_scale=1.0,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, center_z)),
        debug_vis=debug_vis,
    )


def build_trampoline_kinematic_targets(
    default_nodal_state_w: torch.Tensor,
    nodal_kinematic_target: torch.Tensor,
    pin_width: float = TRAMPOLINE_PIN_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create kinematic targets that pin the outer rim of a deformable trampoline."""
    targets = nodal_kinematic_target.clone()
    targets[..., :3] = default_nodal_state_w[..., :3]
    targets[..., 3] = 1.0

    nodal_pos = default_nodal_state_w[..., :3]
    center_xy = nodal_pos[..., :2].mean(dim=1, keepdim=True)
    radial_distance = torch.linalg.vector_norm(nodal_pos[..., :2] - center_xy, dim=-1)

    rim_radius = radial_distance.max(dim=1, keepdim=True).values
    pin_threshold = torch.clamp(rim_radius - pin_width, min=0.0)
    pinned_mask = radial_distance >= pin_threshold
    center_node_ids = radial_distance.argmin(dim=1)

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


def get_trampoline_youngs_moduli(material_view) -> torch.Tensor:
    """Read Young's modulus values from the available material view API."""
    getter = getattr(material_view, "get_youngs_moduli", None)
    if getter is not None:
        values = getter()
    else:
        values = material_view.get_youngs_modulus()
    return torch.as_tensor(values, device="cpu", dtype=torch.float32).clone()


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


def set_trampoline_youngs_moduli(material_view, values: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write Young's modulus values using whichever material-view API is available."""
    env_ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1).contiguous()

    setter = getattr(material_view, "set_youngs_moduli", None)
    if setter is not None:
        values = _as_column_tensor(values)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} Young's modulus values, got {values.shape[0]}.")
        setter(values, indices=env_ids)
    else:
        current_values = _as_column_tensor(material_view.get_youngs_modulus()).clone()
        env_ids = env_ids.to(device=current_values.device)
        values = _as_column_tensor(values, device=current_values.device)
        if values.shape[0] == 1 and env_ids.numel() > 1:
            values = values.expand(env_ids.numel(), 1).clone()
        if values.shape[0] != env_ids.numel():
            raise ValueError(f"Expected {env_ids.numel()} Young's modulus values, got {values.shape[0]}.")
        current_values[env_ids] = values
        material_view.set_youngs_modulus(current_values, indices=env_ids)


__all__ = [
    "TRAMPOLINE_CENTER_Z",
    "TRAMPOLINE_DR_MASS_RANGE",
    "TRAMPOLINE_DR_YOUNGS_MODULUS_RANGE",
    "TRAMPOLINE_MASS",
    "TRAMPOLINE_PIN_WIDTH",
    "TRAMPOLINE_RADIUS",
    "TRAMPOLINE_SIM_RESOLUTION",
    "TRAMPOLINE_THICKNESS",
    "TRAMPOLINE_TOP_Z",
    "TRAMPOLINE_YOUNGS_MODULUS",
    "build_trampoline_kinematic_targets",
    "build_trampoline_visual_translate_ops",
    "get_trampoline_youngs_moduli",
    "make_trampoline_cfg",
    "reset_deformable_trampoline",
    "set_trampoline_youngs_moduli",
    "trampoline_center_heights",
    "trampoline_mesh_prim_path",
    "update_trampoline_visual_height",
]
