# Toroid Generator with ASE

Advanced script for generating toroid (doughnut-shaped) structures from crystalline unit cells using the Atomic Simulation Environment (ASE) library.

## Overview

This repository provides a consolidated, well-documented solution for creating toroid nanostructures from periodic crystal structures. The script combines the best features from multiple previous versions and implements advanced duplicate removal using neighbor list detection.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated functions for each step
- **Flexible Parameters**: Customizable major/minor radii, overlap tolerance, and more
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
| `--input` | `-i` | `Li2O2.cif` | Input CIF file with unit cell structure |
| `--output` | `-o` | `toroid_output.xyz` | Output XYZ file for generated toroid |
| `--R_target` | `-R` | `6.0` | Major radius of toroid in Å |
| `--r_cyl` | `-r` | `2.5` | Minor radius (tube radius) in Å |
| `--overlap_tolerance` | `-t` | `1.0` | Distance threshold for duplicate removal in Å |
| `--prerelax` | | `False` | Enable pre-relaxation with Lennard-Jones |
| `--lj_epsilon` | | `0.002` | LJ epsilon parameter in eV |
| `--lj_sigma` | | `2.5` | LJ sigma parameter in Å |
| `--prerelax_steps` | | `200` | Maximum steps for pre-relaxation |

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

You can also import and use the functions in your own Python scripts:

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

## Legacy Scripts

This repository also contains the original scripts that were consolidated:

- `Toroide.py` - Original version with simple seam padding
- `Toroide_R6_r2.5_New.py` - Version with improved geometric closure
- `Toroide_R6_r2.5_corregido_Version3.py` - Version with neighbor list duplicate removal

The new `toroid_generator.py` combines the best features from all three versions.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately. 
