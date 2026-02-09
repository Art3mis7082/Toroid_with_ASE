# Toroid Generator with ASE

Advanced script for generating toroid (doughnut-shaped) structures from crystalline unit cells using the Atomic Simulation Environment (ASE) library.

## Overview

This repository provides a consolidated, well-documented solution for creating toroid nanostructures from periodic crystal structures. The implementation uses a modular architecture with separate components for core functionality, relaxation, and command-line interface.

### Module Structure

- **`toroid_core.py`**: Core functions for toroid generation (unit cell loading, supercell creation, cylinder extraction, toroid mapping, duplicate removal)
- **`toroid_relaxation.py`**: Pre-relaxation functionality using Lennard-Jones potential
- **`toroid_generator.py`**: Main script with command-line interface and workflow orchestration

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated functions for each step
- **Flexible Parameters**: Customizable major/minor radius, overlap tolerance, and more
- **Advanced Duplicate Removal**: Uses ASE neighbor list for precise seam closure
- **Optional Pre-relaxation**: Lennard-Jones potential to remove atomic clashes
- **Geometric Perfection**: Ensures proper closure using L_target for mapping
- **Command-line Interface**: Easy to use from terminal with argument parsing
- **Comprehensive Documentation**: Detailed docstrings and comments throughout

## Installation

### Requirements

- Python 3.6+
- ASE (Atomic Simulation Environment)
- NumPy

Install dependencies:
```bash
pip install ase numpy
```

## Usage

### Using Configuration File

The easiest way to customize parameters is through a configuration file:

1. Edit `toroid_config.ini` to set your desired parameters:
```ini
[geometry]
R_target = 8.0
r_cyl = 4.0
overlap_tolerance = 1.0

[input_output]
input_file = my_structure.cif
output_file = my_toroid.xyz

[relaxation]
enable_prerelax = true
lj_epsilon = 0.002
lj_sigma = 2.5
prerelax_steps = 200
```

2. Run the generator:
```bash
python toroid_generator.py
```

You can override config file settings with command-line arguments:
```bash
python toroid_generator.py --R_target 10.0 --prerelax
```

Or use a different config file:
```bash
python toroid_generator.py --config my_config.ini
```

### Basic Usage

Generate a toroid from a CIF file with default parameters:
```bash
python toroid_generator.py --input Li2O2.cif --output my_toroid.xyz
```

### Custom Parameters

Specify custom major radius (R) and minor radius (r):
```bash
python toroid_generator.py --input structure.cif --output toroid.xyz --R_target 8.0 --r_cyl 4.0
```

### With Pre-relaxation

Enable Lennard-Jones pre-relaxation to remove atomic overlaps:
```bash
python toroid_generator.py --input Li2O2.cif --output toroid.xyz --prerelax
```

### Full Example with All Parameters

```bash
python toroid_generator.py \
    --input Li2O2.cif \
    --output toroid_relaxed.xyz \
    --R_target 6.0 \
    --r_cyl 2.5 \
    --overlap_tolerance 1.0 \
    --prerelax \
    --lj_epsilon 0.002 \
    --lj_sigma 2.5 \
    --prerelax_steps 200
```

### Command-line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--config` | `-c` | `toroid_config.ini` | Configuration file path |
| `--input` | `-i` | From config | Input CIF file with unit cell structure |
| `--output` | `-o` | From config | Output XYZ file for generated toroid |
| `--R_target` | `-R` | From config | Major radius of toroid in Å |
| `--r_cyl` | `-r` | From config | Minor radius (tube radius) in Å |
| `--overlap_tolerance` | `-t` | From config | Distance threshold for duplicate removal in Å |
| `--prerelax` | | From config | Enable pre-relaxation with Lennard-Jones |
| `--lj_epsilon` | | From config | LJ epsilon parameter in eV |
| `--lj_sigma` | | From config | LJ sigma parameter in Å |
| `--prerelax_steps` | | From config | Maximum steps for pre-relaxation |

## How It Works

The toroid generation process follows these steps:

1. **Read Unit Cell**: Load crystal structure from CIF file
2. **Create Supercell**: Build a block large enough to form the desired toroid
3. **Extract Cylinder**: Filter atoms within cylindrical radius
4. **Map to Toroid**: Apply parametric transformation from cylinder to toroid
5. **Remove Duplicates**: Use neighbor list to identify and remove overlapping atoms at seam
6. **Pre-relax (Optional)**: Apply Lennard-Jones potential to remove clashes
7. **Save Output**: Write final structure to XYZ file

### Toroid Geometry

The toroid is parameterized by:
- **R (R_target)**: Major radius - distance from center of hole to center of tube
- **r (r_cyl)**: Minor radius - radius of the tube itself

The parametric equations used are:
```
X = (R + r*cos(φ)) * cos(θ)
Y = (R + r*cos(φ)) * sin(θ)
Z = r * sin(φ)
```

where θ is the angle around the major circle and φ is the angle around the minor circle.

## Using as a Python Module

You can import and use the functions in your own Python scripts:

### High-level Interface

```python
from toroid_generator import generate_toroid

# Generate a toroid programmatically
toroid = generate_toroid(
    input_file='my_structure.cif',
    output_file='my_toroid.xyz',
    R_target=6.0,
    r_cyl=2.5,
    overlap_tolerance=1.0,
    do_prerelax=True
)

# The returned object is an ASE Atoms object
print(f"Generated toroid with {len(toroid)} atoms")
```

### Low-level Interface

For more control, use the core functions directly:

```python
from toroid_core import (
    read_unit_cell,
    create_cylinder_block,
    extract_cylinder,
    map_cylinder_to_toroid,
    remove_duplicate_atoms
)
from toroid_relaxation import prerelax_structure
import numpy as np

# Step-by-step generation with custom processing
atoms0, (a_x, a_y, a_z) = read_unit_cell('structure.cif')
R_target = 6.0
r_cyl = 2.5
L_target = 2.0 * np.pi * R_target

block = create_cylinder_block(atoms0, a_x, a_y, a_z, L_target, r_cyl, 1.0)
cyl_symbols, cyl_positions = extract_cylinder(block, r_cyl)
toroid_symbols, toroid_positions = map_cylinder_to_toroid(
    cyl_symbols, cyl_positions, R_target, r_cyl, L_target, 1.0
)
toroid = remove_duplicate_atoms(toroid_symbols, toroid_positions, 1.0)

# Optional: apply custom post-processing here
toroid = prerelax_structure(toroid, 0.002, 2.5, 200)
```

## API Reference

### Core Functions (`toroid_core.py`)

#### `read_unit_cell(input_file)`
Read and prepare unit cell from CIF file.
- **Parameters**: `input_file` (str) - Path to CIF file
- **Returns**: `(atoms, (a_x, a_y, a_z))` - ASE Atoms object and lattice parameters

#### `create_cylinder_block(atoms0, a_x, a_y, a_z, L_target, r_cyl, overlap_tolerance)`
Create supercell block for cylinder generation.
- **Parameters**: 
  - `atoms0` - Unit cell structure
  - `a_x, a_y, a_z` - Lattice parameters [Å]
  - `L_target` - Target circumference (2π*R) [Å]
  - `r_cyl` - Cylinder radius [Å]
  - `overlap_tolerance` - Overlap zone width [Å]
- **Returns**: Supercell block structure

#### `extract_cylinder(block, r_cyl)`
Extract atoms within cylindrical radius from block.
- **Parameters**: 
  - `block` - Supercell block
  - `r_cyl` - Cylinder radius [Å]
- **Returns**: `(symbols, positions)` - Chemical symbols and atomic positions

#### `map_cylinder_to_toroid(cyl_symbols, cyl_positions, R_target, r_cyl, L_target, overlap_tolerance)`
Map cylindrical coordinates to toroidal geometry using parametric equations:
- X = (R + r*cos(φ)) * cos(θ)
- Y = (R + r*cos(φ)) * sin(θ)
- Z = r * sin(φ)

**Parameters**: 
- `cyl_symbols, cyl_positions` - Cylinder atoms
- `R_target` - Major radius [Å]
- `r_cyl` - Minor radius [Å]
- `L_target` - Target circumference [Å]
- `overlap_tolerance` - Overlap zone width [Å]

**Returns**: `(symbols, positions)` - Toroid atoms

#### `remove_duplicate_atoms(symbols, positions, overlap_tolerance)`
Remove duplicate atoms at toroid seam using neighbor list detection.
- **Parameters**: 
  - `symbols` - Chemical symbols
  - `positions` - Atomic positions
  - `overlap_tolerance` - Distance threshold for duplicates [Å]
- **Returns**: ASE Atoms object with duplicates removed

### Relaxation Functions (`toroid_relaxation.py`)

#### `prerelax_structure(atoms, epsilon, sigma, max_steps)`
Pre-relax structure using Lennard-Jones potential to remove atomic clashes.
- **Parameters**: 
  - `atoms` - Structure to relax
  - `epsilon` - LJ epsilon parameter [eV]
  - `sigma` - LJ sigma parameter [Å]
  - `max_steps` - Maximum optimization steps
- **Returns**: Relaxed structure
- **Note**: This is NOT chemically accurate; use DFT/MD for accurate simulations

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately. 

This code is implemented with Atomic Simulation Enviroment (ASE) with the following cite:
    - Ask Hjorth Larsen, Jens Jørgen Mortensen, Jakob Blomqvist,
Ivano E. Castelli, Rune Christensen, Marcin Dułak, Jesper Friis,
Michael N. Groves, Bjørk Hammer, Cory Hargus, Eric D. Hermes,
Paul C. Jennings, Peter Bjerre Jensen, James Kermode, John R. Kitchin,
Esben Leonhard Kolsbjerg, Joseph Kubal, Kristen Kaasbjerg,
Steen Lysgaard, Jón Bergmann Maronsson, Tristan Maxson, Thomas Olsen,
Lars Pastewka, Andrew Peterson, Carsten Rostgaard, Jakob Schiøtz,
Ole Schütt, Mikkel Strange, Kristian S. Thygesen, Tejs Vegge,
Lasse Vilhelmsen, Michael Walter, Zhenhua Zeng, Karsten Wedel Jacobsen
The Atomic Simulation Environment—A Python library for working with atoms
J. Phys.: Condens. Matter Vol. 29 273002, 2017

    - A.   Jain, S.P. Ong, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson, 
    The Materials Project: A materials genome approach to accelerating materials innovation. APL Materials, 2013, 1(1), 011002. DOI: 10.17188/1272612

