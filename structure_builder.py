"""Voxel-based constructive geometry for FIB simulations.

All public geometry methods accept physical coordinates in nanometres.  The
result uses ``material_id == -1`` for vacuum and stores the number of atoms in
each occupied voxel.  The current solver is a height-field solver, so build()
validates that every occupied column is connected to z=0 unless explicitly
asked not to do so.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from materials import Material, MaterialRegistry


VACUUM_ID = -1
CoordinateSelector = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class StructureState:
    """Initial material and atom fields consumed by ``simu_funcs.main``."""

    material_id: np.ndarray
    atom_count: np.ndarray
    spacing_nm: tuple[float, float, float]
    materials: MaterialRegistry

    def __post_init__(self) -> None:
        self.material_id = np.asarray(self.material_id, dtype=np.int16)
        self.atom_count = np.asarray(self.atom_count, dtype=np.float32)
        if self.material_id.shape != self.atom_count.shape:
            raise ValueError("material_id and atom_count must have identical shapes")
        if self.material_id.ndim != 3:
            raise ValueError("structure arrays must have shape (NX, NY, NZ)")
        if len(self.spacing_nm) != 3 or min(self.spacing_nm) <= 0.0:
            raise ValueError("spacing_nm must contain three positive values")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.material_id.shape

    @property
    def cell_volume_nm3(self) -> float:
        return float(np.prod(self.spacing_nm))

    def copy(self) -> "StructureState":
        """Return an independent copy suitable for a separate simulation run."""
        return StructureState(
            self.material_id.copy(), self.atom_count.copy(),
            self.spacing_nm, self.materials,
        )

    def validate(self, require_height_field: bool = True, atol: float = 1.0e-5) -> None:
        """Check material IDs, occupancy agreement and per-material capacity."""
        known_ids = {material.material_id for material in self.materials}
        present_ids = set(np.unique(self.material_id).tolist())
        unknown = present_ids.difference(known_ids | {VACUUM_ID})
        if unknown:
            raise ValueError(f"Structure contains unknown material ids: {sorted(unknown)}")
        if np.any(self.atom_count < -atol):
            raise ValueError("atom_count contains negative values")
        occupied_by_id = self.material_id != VACUUM_ID
        occupied_by_atoms = self.atom_count > atol
        if not np.array_equal(occupied_by_id, occupied_by_atoms):
            raise ValueError("material_id occupancy and atom_count occupancy disagree")

        capacities = self.materials.capacity_array(self.cell_volume_nm3)
        occupied = np.nonzero(occupied_by_id)
        if occupied[0].size:
            local_capacity = capacities[self.material_id[occupied]]
            if np.any(self.atom_count[occupied] > local_capacity * (1.0 + 1.0e-5)):
                raise ValueError("one or more voxels exceed their material atom capacity")
        if require_height_field:
            self.validate_height_field()

    def validate_height_field(self) -> None:
        """Reject floating solids, buried voids, bridges and overhangs."""
        occupied = self.material_id != VACUUM_ID
        material_seen = np.maximum.accumulate(occupied, axis=2)
        void_after_material = material_seen & ~occupied
        occupied_after_void = np.maximum.accumulate(void_after_material, axis=2) & occupied
        floating_from_base = occupied[:, :, 0] == 0
        floating_from_base &= np.any(occupied, axis=2)
        invalid_columns = np.any(occupied_after_void, axis=2) | floating_from_base
        if np.any(invalid_columns):
            first = tuple(int(v) for v in np.argwhere(invalid_columns)[0])
            raise ValueError(
                "Structure is not representable by the current height-field solver; "
                f"column {first} contains a buried void, floating solid or overhang"
            )

    def surface_index(self, atom_epsilon: float = 1.0e-6) -> np.ndarray:
        """Return the highest occupied z-index in every lateral column."""
        occupied = self.atom_count > atom_epsilon
        has_material = occupied.any(axis=2)
        reverse = np.argmax(occupied[:, :, ::-1], axis=2)
        surface = self.shape[2] - 1 - reverse
        surface[~has_material] = -1
        return surface.astype(np.int32)

    def surface_material(self, atom_epsilon: float = 1.0e-6) -> np.ndarray:
        """Return the material ID at the top of every nonempty column."""
        surface = self.surface_index(atom_epsilon)
        result = np.full(surface.shape, VACUUM_ID, dtype=np.int16)
        x, y = np.nonzero(surface >= 0)
        result[x, y] = self.material_id[x, y, surface[x, y]]
        return result

    def surface_height_nm(self, atom_epsilon: float = 1.0e-6) -> np.ndarray:
        """Return continuous top height, including a partially filled voxel."""
        surface = self.surface_index(atom_epsilon)
        result = np.zeros(surface.shape, dtype=np.float64)
        x, y = np.nonzero(surface >= 0)
        if x.size:
            z = surface[x, y]
            ids = self.material_id[x, y, z]
            capacities = self.materials.capacity_array(self.cell_volume_nm3)
            fraction = np.clip(self.atom_count[x, y, z] / capacities[ids], 0.0, 1.0)
            result[x, y] = (z + fraction) * self.spacing_nm[2]
        return result

    def summary(self) -> dict:
        """Return JSON-compatible geometry and per-material volume statistics."""
        result = {
            "shape": list(self.shape),
            "spacing_nm": list(self.spacing_nm),
            "domain_nm": [self.shape[i] * self.spacing_nm[i] for i in range(3)],
            "materials": {},
        }
        voxel_volume = self.cell_volume_nm3
        for material in self.materials:
            mask = self.material_id == material.material_id
            capacity = material.atom_capacity(voxel_volume)
            equivalent_voxels = float(self.atom_count[mask].sum(dtype=np.float64) / capacity)
            result["materials"][material.name] = {
                "material_id": material.material_id,
                "occupied_voxels": int(mask.sum()),
                "equivalent_volume_nm3": equivalent_voxels * voxel_volume,
            }
        return result

    def save_npz(self, filename: str | Path) -> None:
        """Persist arrays plus material definitions in a compressed NPZ file."""
        path = Path(filename)
        metadata = {
            "spacing_nm": list(self.spacing_nm),
            "materials": self.materials.to_list(),
        }
        np.savez_compressed(
            path,
            material_id=self.material_id,
            atom_count=self.atom_count,
            metadata_json=np.asarray(json.dumps(metadata)),
        )


def load_structure_npz(filename: str | Path) -> StructureState:
    """Load a structure written by :meth:`StructureState.save_npz`."""
    with np.load(Path(filename), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        materials = MaterialRegistry(
            Material(**values) for values in metadata["materials"]
        )
        state = StructureState(
            data["material_id"].copy(),
            data["atom_count"].copy(),
            tuple(metadata["spacing_nm"]),
            materials,
        )
    state.validate(require_height_field=True)
    return state


class StructureBuilder:
    """Construct a height-field-compatible voxel structure with CSG operations."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        spacing_nm: tuple[float, float, float],
        materials: MaterialRegistry,
    ) -> None:
        if len(shape) != 3 or any(int(value) != value or value <= 0 for value in shape):
            raise ValueError("shape must contain three positive integers")
        if len(spacing_nm) != 3 or min(spacing_nm) <= 0.0:
            raise ValueError("spacing_nm must contain three positive values")
        self.shape = tuple(int(value) for value in shape)
        self.spacing_nm = tuple(float(value) for value in spacing_nm)
        self.materials = materials
        self.material_id = np.full(self.shape, VACUUM_ID, dtype=np.int16)
        self.atom_count = np.zeros(self.shape, dtype=np.float32)
        self._capacity = materials.capacity_array(float(np.prod(self.spacing_nm)))

    @property
    def domain_nm(self) -> tuple[float, float, float]:
        return tuple(self.shape[i] * self.spacing_nm[i] for i in range(3))

    def coordinate_grid_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return x/y coordinates at lateral voxel centers in nm."""
        x = (np.arange(self.shape[0]) + 0.5) * self.spacing_nm[0]
        y = (np.arange(self.shape[1]) + 0.5) * self.spacing_nm[1]
        return np.meshgrid(x, y, indexing="ij")

    def coordinate_grid_xyz(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return broadcastable x/y/z voxel-centre coordinates in nm.

        The returned arrays have shapes ``(NX, 1, 1)``, ``(1, NY, 1)`` and
        ``(1, 1, NZ)``.  A selector can therefore combine them without first
        allocating three full coordinate volumes.
        """
        x = ((np.arange(self.shape[0]) + 0.5) * self.spacing_nm[0])[:, None, None]
        y = ((np.arange(self.shape[1]) + 0.5) * self.spacing_nm[1])[None, :, None]
        z = ((np.arange(self.shape[2]) + 0.5) * self.spacing_nm[2])[None, None, :]
        return x, y, z

    def _axis_mask(self, axis: int, lower_nm: float, upper_nm: float) -> np.ndarray:
        lower = float(lower_nm)
        upper = float(upper_nm)
        if not 0.0 <= lower < upper <= self.domain_nm[axis]:
            raise ValueError(
                f"axis-{axis} bounds [{lower:g}, {upper:g}) nm are outside "
                f"[0, {self.domain_nm[axis]:g}] nm"
            )
        centers = (np.arange(self.shape[axis]) + 0.5) * self.spacing_nm[axis]
        return (centers >= lower) & (centers < upper)

    def _box_mask(
        self,
        x_min_nm: float,
        x_max_nm: float,
        y_min_nm: float,
        y_max_nm: float,
        z_min_nm: float,
        z_max_nm: float,
    ) -> np.ndarray:
        mx = self._axis_mask(0, x_min_nm, x_max_nm)
        my = self._axis_mask(1, y_min_nm, y_max_nm)
        mz = self._axis_mask(2, z_min_nm, z_max_nm)
        return mx[:, None, None] & my[None, :, None] & mz[None, None, :]

    def _fill_mask(self, mask: np.ndarray, material: int | str | Material, overwrite: bool) -> None:
        item = self.materials.get(material)
        if not overwrite and np.any(mask & (self.material_id != VACUUM_ID)):
            raise ValueError("new solid overlaps existing material; pass overwrite=True to replace it")
        self.material_id[mask] = item.material_id
        self.atom_count[mask] = self._capacity[item.material_id]

    def add_box(
        self,
        material: int | str | Material,
        x_min_nm: float,
        x_max_nm: float,
        y_min_nm: float,
        y_max_nm: float,
        z_min_nm: float,
        z_max_nm: float,
        *,
        overwrite: bool = False,
    ) -> "StructureBuilder":
        """Fill a half-open physical-coordinate box with one material."""
        mask = self._box_mask(
            x_min_nm, x_max_nm, y_min_nm, y_max_nm, z_min_nm, z_max_nm
        )
        self._fill_mask(mask, material, overwrite)
        return self

    def add_layer_region(
        self,
        material: int | str | Material,
        thickness_nm: float,
        *,
        x_range_nm: tuple[float, float] | None = None,
        y_range_nm: tuple[float, float] | None = None,
        z_bottom_nm: float = 0.0,
        overwrite: bool = False,
    ) -> "StructureBuilder":
        """Add a constant-thickness material layer in a selected XY region.

        ``x_range_nm`` and ``y_range_nm`` use half-open physical-coordinate
        intervals.  Omitting either range selects that complete domain axis.
        ``z_bottom_nm`` locates the bottom of the layer and
        ``z_bottom_nm + thickness_nm`` locates its top.
        """
        thickness = float(thickness_nm)
        if thickness <= 0.0:
            raise ValueError("thickness_nm must be positive")
        x_bounds = (0.0, self.domain_nm[0]) if x_range_nm is None else x_range_nm
        y_bounds = (0.0, self.domain_nm[1]) if y_range_nm is None else y_range_nm
        if len(x_bounds) != 2 or len(y_bounds) != 2:
            raise ValueError("x_range_nm and y_range_nm must each contain two values")
        return self.add_box(
            material,
            x_bounds[0],
            x_bounds[1],
            y_bounds[0],
            y_bounds[1],
            z_bottom_nm,
            float(z_bottom_nm) + thickness,
            overwrite=overwrite,
        )

    def remove_region(
        self,
        *,
        x_range_nm: tuple[float, float] | None = None,
        y_range_nm: tuple[float, float] | None = None,
        z_range_nm: tuple[float, float] | None = None,
        selector: CoordinateSelector | None = None,
    ) -> "StructureBuilder":
        """Remove material selected by coordinate ranges and/or a function.

        Every supplied condition is combined with logical AND.  A selector is
        called as ``selector(x_nm, y_nm, z_nm)`` with broadcastable voxel-centre
        arrays and must return a boolean-compatible scalar or array.  At least
        one range or selector is required, preventing accidental removal of the
        complete framework.

        The current FIB solver is a height-field solver.  Consequently a custom
        selector should normally remove material from the exposed top downward;
        :meth:`build` rejects buried voids and floating material by default.
        """
        if all(value is None for value in (x_range_nm, y_range_nm, z_range_nm, selector)):
            raise ValueError("provide at least one coordinate range or selector")

        mask = np.ones(self.shape, dtype=bool)
        for axis, bounds in enumerate((x_range_nm, y_range_nm, z_range_nm)):
            if bounds is not None:
                if len(bounds) != 2:
                    raise ValueError("each coordinate range must contain two values")
                axis_mask = self._axis_mask(axis, bounds[0], bounds[1])
                reshape = [1, 1, 1]
                reshape[axis] = self.shape[axis]
                mask &= axis_mask.reshape(reshape)

        if selector is not None:
            if not callable(selector):
                raise TypeError("selector must be callable")
            selected = np.asarray(selector(*self.coordinate_grid_xyz()), dtype=bool)
            try:
                selected = np.broadcast_to(selected, self.shape)
            except ValueError as exc:
                raise ValueError(
                    f"selector result shape {selected.shape} cannot broadcast to {self.shape}"
                ) from exc
            mask &= selected

        self.material_id[mask] = VACUUM_ID
        self.atom_count[mask] = 0.0
        return self

    def fill_below_height_map(
        self,
        material: int | str | Material,
        height_nm: np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        base_z_nm: float = 0.0,
        overwrite: bool = False,
    ) -> "StructureBuilder":
        """Fill material below an array/function-defined continuous surface."""
        item = self.materials.get(material)
        x, y = self.coordinate_grid_xy()
        heights = height_nm(x, y) if callable(height_nm) else np.asarray(height_nm)
        heights = np.broadcast_to(np.asarray(heights, dtype=np.float64), self.shape[:2])
        if np.any(~np.isfinite(heights)):
            raise ValueError("height map contains non-finite values")
        if np.any(heights < base_z_nm) or np.any(heights > self.domain_nm[2]):
            raise ValueError("height map lies outside the z domain")

        dz = self.spacing_nm[2]
        z_bottom = np.arange(self.shape[2], dtype=np.float64) * dz
        fraction = np.clip(
            (heights[:, :, None] - np.maximum(z_bottom[None, None, :], base_z_nm)) / dz,
            0.0,
            1.0,
        )
        # Exclude cells entirely below a nonzero base.
        fraction[:, :, z_bottom + dz <= base_z_nm] = 0.0
        mask = fraction > 0.0
        if not overwrite and np.any(mask & (self.material_id != VACUUM_ID)):
            raise ValueError("height-map solid overlaps existing material")
        self.material_id[mask] = item.material_id
        self.atom_count[mask] = (
            fraction[mask] * self._capacity[item.material_id]
        ).astype(np.float32)
        return self

    def build(self, *, require_height_field: bool = True) -> StructureState:
        """Freeze the mutable builder into a validated simulation state."""
        state = StructureState(
            self.material_id.copy(),
            self.atom_count.copy(),
            self.spacing_nm,
            self.materials,
        )
        state.validate(require_height_field=require_height_field)
        return state


def create_framework(
    shape: tuple[int, int, int],
    spacing_nm: tuple[float, float, float],
    materials: MaterialRegistry,
) -> StructureBuilder:
    """Create an empty simulation framework ready for constructive operations.

    This named entry point mirrors the terminology used by the original model;
    it returns a :class:`StructureBuilder` whose complete domain is initially
    vacuum.
    """
    return StructureBuilder(shape, spacing_nm, materials)


def save_structure_preview(
    state: StructureState,
    output_dir: str | Path,
    *,
    prefix: str = "initial_structure",
) -> None:
    """Save XY height/material maps and central XZ/YZ material cross-sections.

    Plotting is kept outside the builder so construction remains usable in
    headless batch jobs and the solver never depends on Matplotlib.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    height = state.surface_height_nm()
    surface_material = state.surface_material()
    ids = sorted(material.material_id for material in state.materials)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(1, len(ids))))
    # CSV material IDs remain stable when incomplete entries are skipped, so
    # the registered IDs need not be consecutive.  Allocate one color slot for
    # every possible ID and leave unused slots gray; BoundaryNorm then remains
    # valid for both sparse built-in IDs and subsequently assigned custom IDs.
    color_table = np.tile(
        np.asarray([0.75, 0.75, 0.75, 1.0]),
        (max(ids) + 2, 1),
    )
    color_table[0] = [1.0, 1.0, 1.0, 1.0]  # material_id == -1: vacuum
    for material_id, color in zip(ids, colors):
        color_table[material_id + 1] = color
    cmap = ListedColormap(color_table)
    boundaries = np.arange(-1.5, max(ids) + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    extent_xy = [0, state.shape[0] * state.spacing_nm[0], 0, state.shape[1] * state.spacing_nm[1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(height.T, origin="lower", extent=extent_xy, cmap="viridis")
    fig.colorbar(im0, ax=axes[0], label="surface height (nm)")
    axes[0].set_title("Surface height")
    im1 = axes[1].imshow(surface_material.T, origin="lower", extent=extent_xy, cmap=cmap, norm=norm)
    fig.colorbar(im1, ax=axes[1], ticks=[-1, *ids], label="material id")
    axes[1].set_title("Surface material")
    for axis in axes:
        axis.set_xlabel("x (nm)")
        axis.set_ylabel("y (nm)")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_xy.png", dpi=150)
    plt.close(fig)

    mid_x = state.shape[0] // 2
    mid_y = state.shape[1] // 2
    extent_xz = [0, state.shape[0] * state.spacing_nm[0], 0, state.shape[2] * state.spacing_nm[2]]
    extent_yz = [0, state.shape[1] * state.spacing_nm[1], 0, state.shape[2] * state.spacing_nm[2]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(state.material_id[:, mid_y, :].T, origin="lower", extent=extent_xz, cmap=cmap, norm=norm, aspect="auto")
    axes[0].set_title(f"XZ section at y-index {mid_y}")
    axes[0].set_xlabel("x (nm)")
    axes[0].set_ylabel("z (nm)")
    axes[1].imshow(state.material_id[mid_x, :, :].T, origin="lower", extent=extent_yz, cmap=cmap, norm=norm, aspect="auto")
    axes[1].set_title(f"YZ section at x-index {mid_x}")
    axes[1].set_xlabel("y (nm)")
    axes[1].set_ylabel("z (nm)")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_sections.png", dpi=150)
    plt.close(fig)
