"""Simple six-step FIB etching case.

Edit only the physical values below. Geometry classes, derived scan variables,
time-step counts and solver configuration are handled by simu_workflow.
"""

import numpy as np

from simu_workflow import (
    add_material_block,
    add_material_layer,
    define_etch_region,
    define_simulation_size,
    fill_below_height_map,
    remove_initial_region,
    run_simulation,
    set_process_parameters,
)


# 1. Determine simulation size.
simulation = define_simulation_size(
    size_nm=(400, 400, 100),
    cell_size_nm=1.0,
    output_dir="output_simple_fib",
)

# 2. Build the initial structure. These commands may be called repeatedly and
# in any required order before define_etch_region().
add_material_layer(simulation, "Al", thickness_nm=80.0)

# Examples (uncomment/edit as needed):
# To use a mathematical surface instead of the flat layer above, comment out
# add_material_layer() and uncomment the following command:
# fill_below_height_map(
#     simulation,
#     "Al",
#     lambda x, y: 75.0 + 5.0 * np.exp(
#         -((x - 200.0) ** 2 + (y - 200.0) ** 2) / 50.0**2
#     ),
# )
# add_material_layer(
#     simulation, "Al", thickness_nm=5.0, z_bottom_nm=80.0,
# )
# add_material_block(
#     simulation,
#     "Al",
#     x_range_nm=(100.0, 200.0),
#     y_range_nm=(100.0, 200.0),
#     z_range_nm=(80.0, 90.0),
# )
# remove_initial_region(
#     simulation,
#     x_range_nm=(140.0, 160.0),
#     y_range_nm=(100.0, 200.0),
#     z_range_nm=(70.0, 100.0),
# )

# 3. Give the selected etching/scan region.
define_etch_region(
    simulation,
    x_range_nm=(50.0, 350.0),
    y_range_nm=(50.0, 350.0),
    pitch_nm=10.0,
)

# 4. Determine process parameters and the number of complete scan cycles.
set_process_parameters(
    simulation,
    scan_cycles=46,
    voltage_kv=30.0,
    beam_current_a=42.0e-12,
    dwell_time_s=9.75e-7,
    beam_sigma_nm=5.0,
    beam_effect_radius_nm=10.0,
    enable_redeposition=True,
    redeposition_rays=32,
    snapshot_count=50,
)

# 5-6. Run all scan cycles and write the results to output_dir.
if __name__ == "__main__":
    run_simulation(simulation)
