# FIB_Etching_Model
This code package aims to simulate the FIB etching process, the effect of Ga ion beam on the solid substrate.
This model used the cellular automaton and Monte-Carlo method.
The mechanism of the program refers to Xi Lin et. al's article ("Characterization and simulation of sputtering etched profile by focused gallium ion beam on GaN substrate", doi: https://doi.org/10.1016/j.matdes.2022.110563) and developed by myself.

The core scripts of this program contains: (necesssary)
1. material_library.py:  read materials data library and provide self-defined materials.
2. materials.py: calculate the parameters for materials.
3. materials_data.json: materials properties lib.
4. structure_builder: the initial structure builder core.
5. simu_funcs.py: the simulation computation core.
6. simulation_workflow.py: config the running process.
7. simu_run.py: main simulation script.

Others:
1. requestment.txt: requested python packages.
2. fib_gui_demo.py: a simple GUI demo, writed by streamlit.
3. README.md: this description.

Filefolders:
1. test_output: a group of test simulation results, in the real simulation, all the .png, .npy and .vtk files will be outputed.

Commonly, only write and run the simu_run.py to start the simulation.
import the commands from simu_workflow:
``` python
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
```
The commands achieved all basic functions in this package, including define the simulation frame, build the initial structure, define the etchi area, set simulation parameters and run the simulation.
The current simu_run.py here is an examples.

All the materials sputtering data are calculated by SRIM & TRIM (http://www.srim.org/)
