"""Material definitions and SRIM-table loading for the FIB structure model.

The numerical solver identifies every occupied voxel by a small integer material
identifier.  This module owns the mapping between those identifiers and physical
properties; geometry construction is deliberately kept in ``structure_builder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


AVOGADRO_CONSTANT_MOL = 6.02214076e23


@dataclass(frozen=True)
class Material:
    """Physical properties used by the reduced-order FIB model.

    ``molar_mass_g_mol`` is the molar mass of one formula unit and
    ``atoms_per_formula`` is the total number of atoms in that formula unit.
    Sputter yields are total target atoms per incident ion.
    """

    # The dictionaries are keyed by accelerating voltage in kV.  Linear
    # interpolation is allowed only between supplied voltage points; silently
    # extrapolating SRIM data would make a material look more certain than it is.

    material_id: int
    name: str
    density_g_cm3: float
    molar_mass_g_mol: float
    atoms_per_formula: float = 1.0
    sputter_yield_by_kv: Mapping[float, float] = field(default_factory=dict)
    emission_power_by_kv: Mapping[float, float] = field(default_factory=dict)
    sigmund_a_by_kv: Mapping[float, float] = field(default_factory=dict)
    sigmund_alpha_by_kv: Mapping[float, float] = field(default_factory=dict)
    sigmund_beta_by_kv: Mapping[float, float] = field(default_factory=dict)
    # Voltage-independent values are used by user-defined materials.  CSV
    # materials leave these as None and continue to use the tables above.
    normal_sputter_yield_constant: float | None = None
    emission_power_constant: float | None = None
    sticking_coefficient: float = 0.8
    sticking_power: float = 2.0
    # Fallback angular-yield law used until angle-resolved tables are available.
    sigmund_a: float = 107.6
    sigmund_alpha: float = 59.9
    # Lateral width of the deposited-energy distribution.  ``None`` preserves
    # the historical isotropic assumption beta == alpha.
    sigmund_beta: float | None = None

    def __post_init__(self) -> None:
        if self.material_id < 0:
            raise ValueError("material_id must be non-negative; -1 is reserved for vacuum")
        if not self.name.strip():
            raise ValueError("material name cannot be empty")
        if self.density_g_cm3 <= 0.0 or self.molar_mass_g_mol <= 0.0:
            raise ValueError("density and molar mass must be positive")
        if self.atoms_per_formula <= 0.0:
            raise ValueError("atoms_per_formula must be positive")
        if not 0.0 <= self.sticking_coefficient <= 1.0:
            raise ValueError("sticking_coefficient must lie in [0, 1]")
        if self.sticking_power < 0.0:
            raise ValueError("sticking_power must be non-negative")
        if self.sigmund_alpha <= 0.0:
            raise ValueError("sigmund_alpha must be positive")
        if self.sigmund_beta is None:
            object.__setattr__(self, "sigmund_beta", float(self.sigmund_alpha))
        elif self.sigmund_beta <= 0.0:
            raise ValueError("sigmund_beta must be positive")
        table_names = (
            "sputter_yield_by_kv",
            "emission_power_by_kv",
            "sigmund_a_by_kv",
            "sigmund_alpha_by_kv",
            "sigmund_beta_by_kv",
        )
        for name in table_names:
            table = getattr(self, name)
            object.__setattr__(
                self,
                name,
                MappingProxyType({float(k): float(v) for k, v in table.items()}),
            )
        if any(value < 0.0 for value in self.sputter_yield_by_kv.values()):
            raise ValueError("sputter yields must be non-negative")
        if any(value <= -1.0 for value in self.emission_power_by_kv.values()):
            raise ValueError("emission cosine powers must be greater than -1")
        if self.normal_sputter_yield_constant is not None:
            value = float(self.normal_sputter_yield_constant)
            if value < 0.0:
                raise ValueError("normal_sputter_yield_constant must be non-negative")
            if self.sputter_yield_by_kv:
                raise ValueError(
                    "provide either a constant sputter yield or a voltage table, not both"
                )
            object.__setattr__(self, "normal_sputter_yield_constant", value)
        elif not self.sputter_yield_by_kv:
            raise ValueError("sputter-yield data cannot be empty")
        if self.emission_power_constant is not None:
            value = float(self.emission_power_constant)
            if value <= -1.0:
                raise ValueError("emission_power_constant must be greater than -1")
            if self.emission_power_by_kv:
                raise ValueError(
                    "provide either a constant emission power or a voltage table, not both"
                )
            object.__setattr__(self, "emission_power_constant", value)
        elif not self.emission_power_by_kv:
            raise ValueError("emission-power data cannot be empty")
        sigmund_tables = (
            self.sigmund_a_by_kv,
            self.sigmund_alpha_by_kv,
            self.sigmund_beta_by_kv,
        )
        if any(sigmund_tables):
            if not all(sigmund_tables):
                raise ValueError("a, alpha and beta voltage tables must all be provided")
            voltage_sets = [
                set(self.sputter_yield_by_kv),
                set(self.emission_power_by_kv),
                *(set(table) for table in sigmund_tables),
            ]
            if not voltage_sets[0] or any(
                values != voltage_sets[0] for values in voltage_sets[1:]
            ):
                raise ValueError("Y0, n, a, alpha and beta tables must use the same voltages")
            if any(value <= 0.0 for table in sigmund_tables for value in table.values()):
                raise ValueError("a, alpha and beta table values must be positive")

    @staticmethod
    def _interpolate(table: Mapping[float, float], voltage_kv: float, label: str) -> float:
        if not table:
            raise KeyError(f"No {label} data are available")
        voltage = float(voltage_kv)
        keys = np.asarray(sorted(table), dtype=np.float64)
        if voltage < keys[0] or voltage > keys[-1]:
            raise ValueError(
                f"{label} voltage {voltage:g} kV is outside available range "
                f"[{keys[0]:g}, {keys[-1]:g}] kV"
            )
        return float(np.interp(voltage, keys, [table[key] for key in keys]))

    def normal_sputter_yield(self, voltage_kv: float) -> float:
        """Return the normal-incidence total sputter yield in atoms/ion."""
        if self.normal_sputter_yield_constant is not None:
            return self.normal_sputter_yield_constant
        return self._interpolate(self.sputter_yield_by_kv, voltage_kv, "sputter-yield")

    def emission_power(self, voltage_kv: float) -> float:
        """Return the cosine-power exponent used for emitted atoms."""
        if self.emission_power_constant is not None:
            return self.emission_power_constant
        return self._interpolate(self.emission_power_by_kv, voltage_kv, "emission-power")

    def sigmund_parameters(self, voltage_kv: float) -> tuple[float, float, float]:
        """Return voltage-specific ``(a, alpha, beta)`` values in angstrom."""
        if self.sigmund_a_by_kv:
            return (
                self._interpolate(self.sigmund_a_by_kv, voltage_kv, "Sigmund-a"),
                self._interpolate(
                    self.sigmund_alpha_by_kv, voltage_kv, "Sigmund-alpha"
                ),
                self._interpolate(
                    self.sigmund_beta_by_kv, voltage_kv, "Sigmund-beta"
                ),
            )
        return (
            float(self.sigmund_a),
            float(self.sigmund_alpha),
            float(self.sigmund_beta),
        )

    def sputter_yield(self, voltage_kv: float, incidence_angle_rad):
        """Return angle-dependent yield using the current Sigmund fallback law.

        This is intentionally a fallback: angle-resolved SRIM/SDTrimSP tables
        should replace it when they are available for a particular material.
        """
        angle = np.asarray(incidence_angle_rad, dtype=np.float64)
        y0 = self.normal_sputter_yield(voltage_kv)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        sigmund_a, sigmund_alpha, sigmund_beta = self.sigmund_parameters(voltage_kv)
        effective_width = np.sqrt(
            sigmund_alpha**2 * cos_angle**2
            + sigmund_beta**2 * sin_angle**2
        )
        result = y0 * sigmund_alpha / effective_width * cos_angle * np.exp(
            sigmund_a**2 / (2.0 * sigmund_alpha**2)
            * (
                1.0
                - sigmund_alpha**2 / effective_width**2 * cos_angle**2
            )
        )
        return np.maximum(result, 0.0)

    def atom_capacity(self, cell_volume_nm3: float) -> float:
        """Return total target atoms in a completely filled voxel.

        The ``atoms_per_formula`` factor is essential for compounds because the
        solver stores target atoms, while the molar mass is per formula unit.
        """
        cell_volume_cm3 = float(cell_volume_nm3) * 1.0e-21
        return (
            self.density_g_cm3
            * cell_volume_cm3
            / self.molar_mass_g_mol
            * AVOGADRO_CONSTANT_MOL
            * self.atoms_per_formula
        )

    def to_dict(self) -> dict:
        return {
            "material_id": self.material_id,
            "name": self.name,
            "density_g_cm3": self.density_g_cm3,
            "molar_mass_g_mol": self.molar_mass_g_mol,
            "atoms_per_formula": self.atoms_per_formula,
            "sputter_yield_by_kv": dict(self.sputter_yield_by_kv),
            "emission_power_by_kv": dict(self.emission_power_by_kv),
            "sigmund_a_by_kv": dict(self.sigmund_a_by_kv),
            "sigmund_alpha_by_kv": dict(self.sigmund_alpha_by_kv),
            "sigmund_beta_by_kv": dict(self.sigmund_beta_by_kv),
            "normal_sputter_yield_constant": self.normal_sputter_yield_constant,
            "emission_power_constant": self.emission_power_constant,
            "sticking_coefficient": self.sticking_coefficient,
            "sticking_power": self.sticking_power,
            "sigmund_a": self.sigmund_a,
            "sigmund_alpha": self.sigmund_alpha,
            "sigmund_beta": self.sigmund_beta,
        }


class MaterialRegistry:
    """Validated lookup table indexed by both material name and integer id."""

    def __init__(self, materials: Iterable[Material]):
        self._by_id: dict[int, Material] = {}
        self._by_name: dict[str, Material] = {}
        for material in materials:
            if material.material_id in self._by_id:
                raise ValueError(f"Duplicate material id {material.material_id}")
            key = material.name.casefold()
            if key in self._by_name:
                raise ValueError(f"Duplicate material name {material.name!r}")
            self._by_id[material.material_id] = material
            self._by_name[key] = material
        if not self._by_id:
            raise ValueError("At least one material is required")

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def get(self, key: int | str | Material) -> Material:
        """Resolve a material by integer ID, case-insensitive name, or object."""
        if isinstance(key, Material):
            registered = self._by_id.get(key.material_id)
            if registered != key:
                raise KeyError(f"Material {key.name!r} is not registered")
            return registered
        if isinstance(key, (int, np.integer)):
            try:
                return self._by_id[int(key)]
            except KeyError as exc:
                raise KeyError(f"Unknown material id {key}") from exc
        try:
            return self._by_name[str(key).casefold()]
        except KeyError as exc:
            raise KeyError(f"Unknown material name {key!r}") from exc

    @property
    def max_id(self) -> int:
        return max(self._by_id)

    def with_material(self, material: Material) -> "MaterialRegistry":
        """Return a new registry containing all current entries plus material.

        Registries are treated as immutable after construction so builders and
        saved states cannot unexpectedly observe a changed material-ID map.
        Duplicate IDs or names are rejected by the normal constructor checks.
        """
        return MaterialRegistry([*self, material])

    def capacity_array(self, cell_volume_nm3: float) -> np.ndarray:
        """Build a dense ``material_id -> atoms-per-voxel`` lookup array."""
        values = np.zeros(self.max_id + 1, dtype=np.float64)
        for material in self:
            values[material.material_id] = material.atom_capacity(cell_volume_nm3)
        return values

    def property_arrays(self, voltage_kv: float, cell_volume_nm3: float) -> dict[str, np.ndarray]:
        """Build arrays consumed by NumPy/Numba inner loops."""
        size = self.max_id + 1
        arrays = {
            "capacity": np.zeros(size, dtype=np.float64),
            "yield0": np.zeros(size, dtype=np.float64),
            "emission_power": np.zeros(size, dtype=np.float64),
            "sticking_coefficient": np.zeros(size, dtype=np.float64),
            "sticking_power": np.zeros(size, dtype=np.float64),
            "sigmund_a": np.zeros(size, dtype=np.float64),
            "sigmund_alpha": np.ones(size, dtype=np.float64),
            "sigmund_beta": np.ones(size, dtype=np.float64),
        }
        for material in self:
            index = material.material_id
            sigmund_a, sigmund_alpha, sigmund_beta = material.sigmund_parameters(
                voltage_kv
            )
            arrays["capacity"][index] = material.atom_capacity(cell_volume_nm3)
            arrays["yield0"][index] = material.normal_sputter_yield(voltage_kv)
            arrays["emission_power"][index] = material.emission_power(voltage_kv)
            arrays["sticking_coefficient"][index] = material.sticking_coefficient
            arrays["sticking_power"][index] = material.sticking_power
            arrays["sigmund_a"][index] = sigmund_a
            arrays["sigmund_alpha"][index] = sigmund_alpha
            arrays["sigmund_beta"][index] = sigmund_beta
        return arrays

    def to_list(self) -> list[dict]:
        return [self._by_id[key].to_dict() for key in sorted(self._by_id)]


def example_materials() -> MaterialRegistry:
    """Compatibility alias for the package material catalogue.

    New code should import :func:`material_library.get_material_library`
    directly.  The local import avoids a module cycle while preserving older
    scripts that imported ``example_materials`` from this module.
    """
    from material_library import get_material_library

    return get_material_library()
