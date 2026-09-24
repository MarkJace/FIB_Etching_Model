"""Public sequential interface for configuring and running a FIB simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

import simu_funcs
from material_library import add_custom_material, get_material_library
from materials import MaterialRegistry
from structure_builder import StructureBuilder, StructureState, create_framework, save_structure_preview


@dataclass
class _SimulationWorkflow:
    """Internal state; user calculation scripts never define a class."""

    shape: tuple[int, int, int]
    spacing_nm: tuple[float, float, float]
    materials: MaterialRegistry
    builder: StructureBuilder
    output_dir: Path
    scan: dict | None = None
    process: dict | None = None
    _prepared_state: StructureState | None = field(default=None, repr=False)
    _prepared_parameters: dict | None = field(default=None, repr=False)


def define_simulation_size(
    *,
    size_nm: tuple[float, float, float],
    cell_size_nm: float | tuple[float, float, float] = 1.0,
    output_dir: str | Path = "output_fib",
) -> _SimulationWorkflow:
    """Step 1: create a physical-size domain and load built-in materials."""
    spacing_nm = (
        (float(cell_size_nm),) * 3
        if np.isscalar(cell_size_nm)
        else tuple(float(value) for value in cell_size_nm)
    )
    if len(size_nm) != 3 or len(spacing_nm) != 3:
        raise ValueError("size_nm and cell_size_nm must define three axes")
    if min(size_nm) <= 0.0 or min(spacing_nm) <= 0.0:
        raise ValueError("simulation and cell sizes must be positive")
    cells = np.asarray(size_nm, dtype=np.float64) / np.asarray(spacing_nm)
    rounded = np.rint(cells)
    if not np.allclose(cells, rounded):
        raise ValueError("each simulation size must be exactly divisible by its cell size")
    shape = tuple(int(value) for value in rounded)
    materials = get_material_library()
    builder = create_framework(shape, spacing_nm, materials)
    return _SimulationWorkflow(
        builder.shape, builder.spacing_nm, materials, builder, Path(output_dir)
    )


def define_custom_material(simulation: _SimulationWorkflow, **parameters) -> None:
    """Optionally register a custom material before adding solid geometry."""
    if np.any(simulation.builder.material_id >= 0):
        raise RuntimeError("define custom materials before building the initial structure")
    simulation.materials = add_custom_material(simulation.materials, **parameters)
    simulation.builder = create_framework(
        simulation.shape, simulation.spacing_nm, simulation.materials
    )
    simulation._prepared_state = None
    simulation._prepared_parameters = None


def add_material_layer(
    simulation: _SimulationWorkflow,
    material: str | int,
    thickness_nm: float,
    *,
    z_bottom_nm: float = 0.0,
    x_range_nm: tuple[float, float] | None = None,
    y_range_nm: tuple[float, float] | None = None,
    overwrite: bool = False,
) -> None:
    """Step 2: add a material layer to all or part of the XY domain."""
    simulation.builder.add_layer_region(
        material,
        thickness_nm,
        x_range_nm=x_range_nm,
        y_range_nm=y_range_nm,
        z_bottom_nm=z_bottom_nm,
        overwrite=overwrite,
    )
    simulation._prepared_state = None
    simulation._prepared_parameters = None


def add_material_block(
    simulation: _SimulationWorkflow,
    material: str | int,
    *,
    x_range_nm: tuple[float, float],
    y_range_nm: tuple[float, float],
    z_range_nm: tuple[float, float],
    overwrite: bool = False,
) -> None:
    """Step 2: add one material-filled rectangular block to the structure.

    All coordinate ranges are half-open physical intervals ``[start, end)`` in
    nm. By default the block may only fill vacuum; use ``overwrite=True`` when
    the block is intentionally replacing existing material.
    """
    ranges = {
        "x_range_nm": x_range_nm,
        "y_range_nm": y_range_nm,
        "z_range_nm": z_range_nm,
    }
    invalid = [name for name, bounds in ranges.items() if len(bounds) != 2]
    if invalid:
        raise ValueError(
            "coordinate ranges must each contain two values: " + ", ".join(invalid)
        )
    simulation.builder.add_box(
        material,
        x_range_nm[0],
        x_range_nm[1],
        y_range_nm[0],
        y_range_nm[1],
        z_range_nm[0],
        z_range_nm[1],
        overwrite=overwrite,
    )
    simulation._prepared_state = None
    simulation._prepared_parameters = None


def fill_below_height_map(
    simulation: _SimulationWorkflow,
    material: str | int,
    height_nm: np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    base_z_nm: float = 0.0,
    overwrite: bool = False,
) -> None:
    """Step 2: fill material below an array/function-defined surface.

    ``height_nm`` is either an ``(NX, NY)`` height array or a function called
    as ``height_nm(x_nm, y_nm)`` on the lateral voxel-centre coordinate grid.
    Heights and ``base_z_nm`` are physical coordinates in nm.
    """
    simulation.builder.fill_below_height_map(
        material,
        height_nm,
        base_z_nm=base_z_nm,
        overwrite=overwrite,
    )
    simulation._prepared_state = None
    simulation._prepared_parameters = None


def remove_initial_region(
    simulation: _SimulationWorkflow,
    *,
    x_range_nm: tuple[float, float] | None = None,
    y_range_nm: tuple[float, float] | None = None,
    z_range_nm: tuple[float, float] | None = None,
    selector: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> None:
    """Optionally remove a range/function-selected part of the initial solid."""
    simulation.builder.remove_region(
        x_range_nm=x_range_nm,
        y_range_nm=y_range_nm,
        z_range_nm=z_range_nm,
        selector=selector,
    )
    simulation._prepared_state = None
    simulation._prepared_parameters = None


def define_etch_region(
    simulation: _SimulationWorkflow,
    *,
    x_range_nm: tuple[float, float],
    y_range_nm: tuple[float, float],
    pitch_nm: float | tuple[float, float] = 10.0,
) -> None:
    """Step 3: define inclusive serpentine-scan endpoints and point pitch."""
    pitch = (
        (float(pitch_nm), float(pitch_nm))
        if np.isscalar(pitch_nm)
        else tuple(pitch_nm)
    )
    if len(x_range_nm) != 2 or len(y_range_nm) != 2 or len(pitch) != 2:
        raise ValueError("etch ranges and pitch must each contain two axis values")
    lx = simulation.shape[0] * simulation.spacing_nm[0]
    ly = simulation.shape[1] * simulation.spacing_nm[1]
    x0, x1 = map(float, x_range_nm)
    y0, y1 = map(float, y_range_nm)
    px, py = map(float, pitch)
    if not 0.0 <= x0 <= x1 <= lx or not 0.0 <= y0 <= y1 <= ly:
        raise ValueError("etch region must lie inside the simulation XY domain")
    if px <= 0.0 or py <= 0.0:
        raise ValueError("scan pitch must be positive")
    simulation.scan = {
        "x_range_nm": (x0, x1),
        "y_range_nm": (y0, y1),
        "pitch_nm": (px, py),
    }
    simulation._prepared_parameters = None


def set_process_parameters(
    simulation: _SimulationWorkflow,
    *,
    scan_cycles: int,
    voltage_kv: float = 30.0,
    beam_current_a: float = 42.0e-12,
    dwell_time_s: float = 9.75e-7,
    beam_sigma_nm: float = 5.0,
    beam_effect_radius_nm: float = 10.0,
    max_time_step_s: float = 5.0e-7,
    return_velocity_nm_s: float = 1.0e6,
    poisson_ion_arrival: bool = True,
    random_seed: int = 20260820,
    enable_redeposition: bool = True,
    redeposition_rays: int = 32,
    snapshot_count: int = 50,
    atom_epsilon: float = 1.0e-6,
    colorbar_mode: str = "global",
) -> None:
    """Step 4: set process inputs, including the number of complete scans."""
    if isinstance(scan_cycles, bool) or not isinstance(scan_cycles, Integral):
        raise TypeError("scan_cycles must be a positive integer")
    if scan_cycles <= 0:
        raise ValueError("scan_cycles must be a positive integer")
    positive = {
        "voltage_kv": voltage_kv,
        "beam_current_a": beam_current_a,
        "dwell_time_s": dwell_time_s,
        "beam_sigma_nm": beam_sigma_nm,
        "beam_effect_radius_nm": beam_effect_radius_nm,
        "max_time_step_s": max_time_step_s,
        "return_velocity_nm_s": return_velocity_nm_s,
        "atom_epsilon": atom_epsilon,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0.0]
    if invalid:
        raise ValueError("parameters must be positive: " + ", ".join(invalid))
    if int(snapshot_count) <= 0:
        raise ValueError("snapshot_count must be positive")
    if enable_redeposition and int(redeposition_rays) <= 0:
        raise ValueError("redeposition_rays must be positive when redeposition is enabled")
    if colorbar_mode not in {"global", "adaptive"}:
        raise ValueError("colorbar_mode must be 'global' or 'adaptive'")
    simulation.process = {
        "scan_cycles": int(scan_cycles),
        "voltage_kv": float(voltage_kv),
        "beam_current_a": float(beam_current_a),
        "dwell_time_s": float(dwell_time_s),
        "beam_sigma_nm": float(beam_sigma_nm),
        "beam_effect_radius_nm": float(beam_effect_radius_nm),
        "max_time_step_s": float(max_time_step_s),
        "return_velocity_nm_s": float(return_velocity_nm_s),
        "poisson_ion_arrival": bool(poisson_ion_arrival),
        "random_seed": int(random_seed),
        "enable_redeposition": bool(enable_redeposition),
        "redeposition_rays": int(redeposition_rays),
        "snapshot_count": int(snapshot_count),
        "atom_epsilon": float(atom_epsilon),
        "colorbar_mode": colorbar_mode,
    }
    simulation._prepared_parameters = None


def _point_count(start: float, end: float, pitch: float, axis: str) -> int:
    count = int(round((end - start) / pitch)) + 1
    if not np.isclose((count - 1) * pitch, end - start):
        raise ValueError(f"pitch must exactly divide the {axis} etch-range length")
    return count


def prepare_simulation(
    simulation: _SimulationWorkflow,
) -> tuple[StructureState, Mapping[str, object]]:
    """Validate setup and calculate the complete solver parameter snapshot."""
    if simulation.scan is None:
        raise RuntimeError("call define_etch_region() before preparing the simulation")
    if simulation.process is None:
        raise RuntimeError("call set_process_parameters() before preparing the simulation")
    state = simulation.builder.build(require_height_field=True)
    if not np.any(state.material_id >= 0):
        raise ValueError("initial structure contains no material")

    nx, ny, nz = simulation.shape
    dx, dy, dz = simulation.spacing_nm
    x0, x1 = simulation.scan["x_range_nm"]
    y0, y1 = simulation.scan["y_range_nm"]
    pitch_x, pitch_y = simulation.scan["pitch_nm"]
    process = simulation.process
    npx = _point_count(x0, x1, pitch_x, "x")
    npy = _point_count(y0, y1, pitch_y, "y")
    points_per_pass = npx * npy
    steps_per_dwell = max(
        1, math.ceil(process["dwell_time_s"] / process["max_time_step_s"])
    )
    dt = process["dwell_time_s"] / steps_per_dwell
    scan_steps = points_per_pass * steps_per_dwell
    final_x = x1 if (npy - 1) % 2 == 0 else x0
    final_y = y1
    return_distance = float(np.hypot(final_x - x0, final_y - y0))
    return_time = return_distance / process["return_velocity_nm_s"]
    return_steps = math.ceil(return_time / dt)
    period_steps = scan_steps + return_steps
    n_pass = process["scan_cycles"]
    full_steps = n_pass * period_steps

    cell_volume_nm3 = dx * dy * dz
    first_material = next(iter(state.materials))
    # Validate that every registered material covers the selected voltage before
    # the expensive simulation or any output writing begins.
    state.materials.property_arrays(process["voltage_kv"], cell_volume_nm3)
    initial_layers = max(1, int(np.max(state.surface_index())) + 1)
    ion_rate = process["beam_current_a"] / 1.6e-19
    scan_time = points_per_pass * process["dwell_time_s"]
    expected_per_point = ion_rate * process["dwell_time_s"]
    expected_per_pass = expected_per_point * points_per_pass
    first_a, first_alpha, first_beta = first_material.sigmund_parameters(
        process["voltage_kv"]
    )

    parameters = {
        "DX_NM": dx, "DY_NM": dy, "DZ_NM": dz,
        "NX": nx, "NY": ny, "NZ": nz,
        "INITIAL_SOLID_LAYERS": initial_layers,
        "ATOM_COUNT_PER_CELL": first_material.atom_capacity(cell_volume_nm3),
        "ATOM_EPSILON": process["atom_epsilon"],
        "ION_RATE_S": ion_rate,
        "BEAM_EFFECT_RADIUS_NM": process["beam_effect_radius_nm"],
        "BEAM_SIGMA_NM": process["beam_sigma_nm"],
        "N_POINTS_X": npx, "N_POINTS_Y": npy,
        "N_POINTS_PER_PASS": points_per_pass,
        "DWELL_TIME_S": process["dwell_time_s"],
        "STEPS_PER_DWELL": steps_per_dwell, "DT_S": dt,
        "N_STEPS": full_steps, "FULL_N_STEPS": full_steps,
        "PASS_PERIOD_STEPS": period_steps, "SCAN_STEPS_PER_PASS": scan_steps,
        "SCAN_X_START_NM": x0, "SCAN_X_END_NM": x1,
        "SCAN_Y_START_NM": y0, "SCAN_Y_END_NM": y1,
        "PITCH_X_NM": pitch_x, "PITCH_Y_NM": pitch_y,
        "SCAN_INITIAL_X_NM": x0, "SCAN_INITIAL_Y_NM": y0,
        "SCAN_FINAL_X_NM": final_x, "SCAN_FINAL_Y_NM": final_y,
        "IB_RETURN_VELOCITY_NM_S": process["return_velocity_nm_s"],
        "IB_RETURN_DISTANCE_NM": return_distance,
        "IB_RETURN_TIME_S": return_time, "IB_RETURN_STEPS": return_steps,
        "N_PASS": n_pass,
        "Y0": first_material.normal_sputter_yield(process["voltage_kv"]),
        "A_SIGMUND": first_a,
        "ALPHA_SIGMUND": first_alpha,
        "BETA_SIGMUND": first_beta,
        "USE_POISSON_ION_ARRIVAL": process["poisson_ion_arrival"],
        "ION_RNG": np.random.default_rng(process["random_seed"]),
        "ENABLE_REDEPOSTION": process["enable_redeposition"],
        "REDEPO_RAYS": process["redeposition_rays"],
        "REDEPO_COS_POWER": 2.0,
        "STICKING_COEFFICIENT": 0.8, "STICKING_POWER": 2.0,
        "REDEPOSITION_RNG": np.random.default_rng(process["random_seed"] + 1),
        "OUTPUT_DIR": str(simulation.output_dir),
        "SNAPSHOT_INTERVAL": max(1, full_steps // process["snapshot_count"]),
        "SURFACE_HEIGHT_PLOT_RANGE_NM": 2.0,
        "COLORBAR_MODE": process["colorbar_mode"],
        "VOLTAGE_KV": process["voltage_kv"],
        "I_BEAM_A": process["beam_current_a"],
        "ELEMENTARY_CHARGE_C": 1.6e-19,
        "DOMAIN_HEIGHT_Z_NM": nz * dz,
        "SCAN_TIME_PER_PASS_S": scan_time,
        "EXPECTED_IONS_PER_POINT": expected_per_point,
        "EXPECTED_IONS_PER_PASS": expected_per_pass,
        "PLANNED_EXPECTED_IONS": expected_per_pass * n_pass,
        "FULL_TOTAL_TIME_S": full_steps * dt,
        "TOTAL_TIME_S": full_steps * dt,
    }
    simulation._prepared_state = state
    simulation._prepared_parameters = parameters
    return state, parameters


def run_simulation(simulation: _SimulationWorkflow) -> Path:
    """Steps 5-6: run the requested complete scan cycles and save results."""
    state, parameters = prepare_simulation(simulation)
    simulation.output_dir.mkdir(parents=True, exist_ok=True)
    state.save_npz(simulation.output_dir / "initial_structure.npz")
    save_structure_preview(state, simulation.output_dir)
    simu_funcs.configure(parameters)
    simu_funcs.main(state)
    return simulation.output_dir
