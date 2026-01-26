#!/usr/bin/env python3
"""
================================================================================
Toroid Generator using ASE (Atomic Simulation Environment)
================================================================================

This script generates toroid (doughnut-shaped) structures from a crystalline
unit cell using the Atomic Simulation Environment (ASE) library.

The process follows these steps:
1. Read a unit cell from a CIF file
2. Create a supercell (block) large enough to form a cylinder
3. Filter atoms to create a cylindrical structure
4. Map the cylinder onto a toroid geometry
5. Remove duplicate atoms at the seam using neighbor list detection
6. Optionally pre-relax the structure using Lennard-Jones potential

Main Parameters:
----------------
- R_target: Major radius of the toroid (center of hole to center of tube) [Å]
- r_cyl: Minor radius (thickness of the toroid ring) [Å]
- overlap_tolerance: Distance threshold for duplicate atom removal [Å]
- input_file: Path to input CIF file with unit cell structure
- output_file: Path to output XYZ file for the generated toroid

The script uses advanced neighbor list detection to ensure proper closure
of the toroid without gaps or overlapping atoms at the seam.

Author: Consolidated from multiple versions
Date: 2026-01-23
"""

import numpy as np
from ase.io import write
import argparse
import sys
import configparser
import os

# Import core functionality from modular components
from toroid_core import (
    read_unit_cell,
    create_cylinder_block,
    extract_cylinder,
    map_cylinder_to_toroid,
    remove_duplicate_atoms
)
from toroid_relaxation import prerelax_structure


def load_config(config_file='toroid_config.ini'):
    """
    Load configuration from INI file.
    
    Parameters:
        config_file (str): Path to configuration file
    
    Returns:
        dict: Configuration parameters
    """
    config = configparser.ConfigParser()
    defaults = {
        'R_target': 6.0,
        'r_cyl': 2.5,
        'overlap_tolerance': 1.0,
        'input_file': 'Li2O2.cif',
        'output_file': 'toroid_output.xyz',
        'enable_prerelax': False,
        'lj_epsilon': 0.002,
        'lj_sigma': 2.5,
        'prerelax_steps': 200
    }
    
    if os.path.exists(config_file):
        config.read(config_file)
        
        # Parse geometry section
        if 'geometry' in config:
            defaults['R_target'] = config.getfloat('geometry', 'R_target', fallback=defaults['R_target'])
            defaults['r_cyl'] = config.getfloat('geometry', 'r_cyl', fallback=defaults['r_cyl'])
            defaults['overlap_tolerance'] = config.getfloat('geometry', 'overlap_tolerance', fallback=defaults['overlap_tolerance'])
        
        # Parse input_output section
        if 'input_output' in config:
            defaults['input_file'] = config.get('input_output', 'input_file', fallback=defaults['input_file'])
            defaults['output_file'] = config.get('input_output', 'output_file', fallback=defaults['output_file'])
        
        # Parse relaxation section
        if 'relaxation' in config:
            defaults['enable_prerelax'] = config.getboolean('relaxation', 'enable_prerelax', fallback=defaults['enable_prerelax'])
            defaults['lj_epsilon'] = config.getfloat('relaxation', 'lj_epsilon', fallback=defaults['lj_epsilon'])
            defaults['lj_sigma'] = config.getfloat('relaxation', 'lj_sigma', fallback=defaults['lj_sigma'])
            defaults['prerelax_steps'] = config.getint('relaxation', 'prerelax_steps', fallback=defaults['prerelax_steps'])
    
    return defaults


def parse_arguments():
    """Parse command-line arguments."""
    # Load config file first to get defaults
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description='Generate toroid structures from crystalline unit cells',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input Li2O2.cif --output toroid.xyz --R_target 6.0 --r_cyl 2.5
  %(prog)s --config my_config.ini
  %(prog)s --input structure.cif --R_target 8.0 --r_cyl 4.0 --prerelax
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='toroid_config.ini',
                        help='Configuration file (default: toroid_config.ini)')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help=f'Input CIF file (default from config: {config["input_file"]})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help=f'Output XYZ file (default from config: {config["output_file"]})')
    parser.add_argument('--R_target', '-R', type=float, default=None,
                        help=f'Major radius in Å (default from config: {config["R_target"]})')
    parser.add_argument('--r_cyl', '-r', type=float, default=None,
                        help=f'Minor radius in Å (default from config: {config["r_cyl"]})')
    parser.add_argument('--overlap_tolerance', '-t', type=float, default=None,
                        help=f'Overlap tolerance in Å (default from config: {config["overlap_tolerance"]})')
    parser.add_argument('--prerelax', action='store_true', default=None,
                        help=f'Enable pre-relaxation (default from config: {config["enable_prerelax"]})')
    parser.add_argument('--lj_epsilon', type=float, default=None,
                        help=f'LJ epsilon in eV (default from config: {config["lj_epsilon"]})')
    parser.add_argument('--lj_sigma', type=float, default=None,
                        help=f'LJ sigma in Å (default from config: {config["lj_sigma"]})')
    parser.add_argument('--prerelax_steps', type=int, default=None,
                        help=f'Max relaxation steps (default from config: {config["prerelax_steps"]})')
    
    args = parser.parse_args()
    
    # Reload config if a different config file was specified
    if args.config != 'toroid_config.ini':
        config = load_config(args.config)
    
    # Override config with command-line arguments (if provided)
    if args.input is not None:
        config['input_file'] = args.input
    if args.output is not None:
        config['output_file'] = args.output
    if args.R_target is not None:
        config['R_target'] = args.R_target
    if args.r_cyl is not None:
        config['r_cyl'] = args.r_cyl
    if args.overlap_tolerance is not None:
        config['overlap_tolerance'] = args.overlap_tolerance
    if args.prerelax is not None:
        config['enable_prerelax'] = args.prerelax
    if args.lj_epsilon is not None:
        config['lj_epsilon'] = args.lj_epsilon
    if args.lj_sigma is not None:
        config['lj_sigma'] = args.lj_sigma
    if args.prerelax_steps is not None:
        config['prerelax_steps'] = args.prerelax_steps
    
    return config


def generate_toroid(input_file='Li2O2.cif', 
                   output_file='toroid_output.xyz',
                   R_target=6.0,
                   r_cyl=2.5,
                   overlap_tolerance=1.0,
                   do_prerelax=False,
                   lj_epsilon=0.002,
                   lj_sigma=2.5,
                   prerelax_steps=200):
    """
    Main function to generate a toroid structure.
    
    Parameters:
        input_file (str): Path to input CIF file
        output_file (str): Path to output XYZ file
        R_target (float): Major radius in Å
        r_cyl (float): Minor radius (tube radius) in Å
        overlap_tolerance (float): Distance threshold for duplicate removal in Å
        do_prerelax (bool): Whether to pre-relax with LJ potential
        lj_epsilon (float): LJ epsilon parameter in eV
        lj_sigma (float): LJ sigma parameter in Å
        prerelax_steps (int): Maximum pre-relaxation steps
    
    Returns:
        Atoms: Generated toroid structure
    """
    print("="*80)
    print("TOROID GENERATOR - ASE")
    print("="*80)
    
    # Validate input parameters
    if R_target <= 0:
        print(f"Error: Major radius (R_target) must be positive, got {R_target}")
        sys.exit(1)
    if r_cyl <= 0:
        print(f"Error: Minor radius (r_cyl) must be positive, got {r_cyl}")
        sys.exit(1)
    if overlap_tolerance <= 0:
        print(f"Error: Overlap tolerance must be positive, got {overlap_tolerance}")
        sys.exit(1)
    
    print(f"\nParameters:")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    print(f"  Major radius (R): {R_target:.3f} Å")
    print(f"  Minor radius (r): {r_cyl:.3f} Å")
    print(f"  Overlap tolerance: {overlap_tolerance:.3f} Å")
    print(f"  Pre-relaxation: {'Yes' if do_prerelax else 'No'}")
    
    # Step 1: Read unit cell
    atoms0, (a_x, a_y, a_z) = read_unit_cell(input_file)
    
    # Step 2: Calculate target circumference (used by multiple steps)
    L_target = 2.0 * np.pi * R_target
    
    # Step 3: Create cylinder block
    block = create_cylinder_block(atoms0, a_x, a_y, a_z, L_target, r_cyl, overlap_tolerance)
    
    # Step 4: Extract cylindrical structure
    cyl_symbols, cyl_positions = extract_cylinder(block, r_cyl)
    
    # Step 5: Map to toroid
    toroid_symbols, toroid_positions = map_cylinder_to_toroid(
        cyl_symbols, cyl_positions, R_target, r_cyl, L_target, overlap_tolerance
    )
    
    # Step 6: Remove duplicates at seam
    toroid = remove_duplicate_atoms(toroid_symbols, toroid_positions, overlap_tolerance)
    
    # Step 7: Optional pre-relaxation
    if do_prerelax:
        toroid = prerelax_structure(toroid, lj_epsilon, lj_sigma, prerelax_steps)
    
    # Step 8: Save output
    write(output_file, toroid)
    print(f"\n{'='*80}")
    print(f"✓ SUCCESS: Toroid saved to {output_file}")
    print(f"  Final atom count: {len(toroid)}")
    print(f"{'='*80}\n")
    
    return toroid


def main():
    """Main entry point for command-line usage."""
    config = parse_arguments()
    
    generate_toroid(
        input_file=config['input_file'],
        output_file=config['output_file'],
        R_target=config['R_target'],
        r_cyl=config['r_cyl'],
        overlap_tolerance=config['overlap_tolerance'],
        do_prerelax=config['enable_prerelax'],
        lj_epsilon=config['lj_epsilon'],
        lj_sigma=config['lj_sigma'],
        prerelax_steps=config['prerelax_steps']
    )


if __name__ == '__main__':
    main()
