"""
Core functions for toroid generation from crystalline unit cells.

This module contains the fundamental operations for creating toroid structures:
- Reading and preparing unit cells
- Creating supercell blocks
- Extracting cylindrical structures
- Mapping cylinders to toroid geometry
- Removing duplicate atoms at seams

Author: Consolidated from multiple versions
Date: 2026-01-23
"""

import numpy as np
import sys
from ase import Atoms
from ase.io import read
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar
from ase.neighborlist import build_neighbor_list


def read_unit_cell(input_file):
    """
    Read and prepare the unit cell from a CIF file.
    
    Parameters:
        input_file (str): Path to the CIF file
    
    Returns:
        tuple: (atoms, cell_parameters)
            - atoms: ASE Atoms object with the unit cell
            - cell_parameters: tuple of (a_x, a_y, a_z) lattice parameters
    """
    try:
        atoms = read(input_file)
        atoms.set_pbc([True, True, True])
        atoms.wrap()
        
        # Extract lattice parameters
        cell_params = cell_to_cellpar(atoms.get_cell())[:3]
        a_x, a_y, a_z = cell_params
        
        print(f"✓ Unit cell loaded from {input_file}")
        print(f"  Lattice parameters: a={a_x:.3f} Å, b={a_y:.3f} Å, c={a_z:.3f} Å")
        print(f"  Atoms in unit cell: {len(atoms)}")
        
        return atoms, (a_x, a_y, a_z)
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)


def create_cylinder_block(atoms0, a_x, a_y, a_z, L_target, r_cyl, overlap_tolerance):
    """
    Create a supercell block large enough to form a cylinder.
    
    The block is sized to:
    - Cover the circumference L_target in the x-direction (plus overlap zone)
    - Extend at least 2*r_cyl in y and z directions
    
    Parameters:
        atoms0 (Atoms): Unit cell structure
        a_x, a_y, a_z (float): Lattice parameters in Å
        L_target (float): Target circumference (2π*R_target) in Å
        r_cyl (float): Cylinder radius in Å
        overlap_tolerance (float): Overlap zone width in Å
    
    Returns:
        Atoms: Supercell block structure
    """
    # Calculate repetitions needed
    # For x: need to cover L_target plus overlap zone
    nx = max(2, int(np.ceil((L_target + overlap_tolerance) / a_x)))
    
    # For y, z: need to cover diameter with safety factor
    ny = max(2, int(np.ceil((2.2 * r_cyl) / a_y)))
    nz = max(2, int(np.ceil((2.2 * r_cyl) / a_z)))
    
    print(f"\n✓ Creating supercell block")
    print(f"  Repetitions: nx={nx}, ny={ny}, nz={nz}")
    print(f"  Target circumference: {L_target:.3f} Å")
    
    # Create supercell
    P = np.diag([nx, ny, nz])
    block = make_supercell(atoms0, P)
    block.set_pbc([False, False, False])  # Work as a cluster
    block.wrap()
    
    print(f"  Atoms in block: {len(block)}")
    
    return block


def extract_cylinder(block, r_cyl):
    """
    Extract atoms within cylindrical radius from the block.
    
    The cylinder axis is along x, with the transverse plane in y-z centered at origin.
    
    Parameters:
        block (Atoms): Supercell block
        r_cyl (float): Cylinder radius in Å
    
    Returns:
        tuple: (symbols, positions)
            - symbols: Chemical symbols of atoms in cylinder
            - positions: Positions of atoms in cylinder
    """
    pos = block.get_positions()
    
    # Get coordinate ranges
    x_min, x_max = pos[:,0].min(), pos[:,0].max()
    y_min, y_max = pos[:,1].min(), pos[:,1].max()
    z_min, z_max = pos[:,2].min(), pos[:,2].max()
    
    # Translate block: x starts at 0, y-z centered at origin
    block.translate([-x_min, -(y_min + y_max)/2.0, -(z_min + z_max)/2.0])
    pos = block.get_positions()
    
    # Filter atoms by radial distance in y-z plane
    y = pos[:,1]
    z = pos[:,2]
    r_perp = np.sqrt(y**2 + z**2)
    keep = r_perp <= r_cyl
    
    cyl_symbols = np.array(block.get_chemical_symbols())[keep]
    cyl_positions = pos[keep]
    
    print(f"\n✓ Cylindrical structure extracted")
    print(f"  Atoms in cylinder: {len(cyl_symbols)}")
    print(f"  Cylinder radius: {r_cyl:.3f} Å")
    
    return cyl_symbols, cyl_positions


def map_cylinder_to_toroid(cyl_symbols, cyl_positions, R_target, r_cyl, L_target, overlap_tolerance):
    """
    Map cylindrical coordinates to toroidal geometry.
    
    The mapping uses:
    - s: position along cylinder axis (x-coordinate)
    - theta = 2π * s / L_target: angle around major circle
    - (r, phi): polar coordinates in transverse plane
    
    Toroid equations:
        X = (R + r*cos(phi)) * cos(theta)
        Y = (R + r*cos(phi)) * sin(theta)
        Z = r * sin(phi)
    
    Parameters:
        cyl_symbols (array): Chemical symbols
        cyl_positions (array): Positions in cylindrical coordinates
        R_target (float): Major radius in Å
        r_cyl (float): Minor radius in Å
        L_target (float): Target circumference in Å
        overlap_tolerance (float): Width of overlap zone in Å
    
    Returns:
        tuple: (symbols, positions)
            - symbols: Chemical symbols (list)
            - positions: 3D positions on toroid (array)
    """
    # Extract coordinates
    s = cyl_positions[:,0] - cyl_positions[:,0].min()  # Normalize to start at 0
    y = cyl_positions[:,1]
    z = cyl_positions[:,2]
    
    # Convert to cylindrical coordinates (transverse)
    phi = np.arctan2(z, y)  # Azimuthal angle in transverse plane
    r = np.sqrt(y**2 + z**2)  # Radial distance from axis
    
    # Map to toroid: angle theta based on position s along original axis
    theta = 2.0 * np.pi * (s / L_target)
    
    # Toroid parametric equations
    X = (R_target + r * np.cos(phi)) * np.cos(theta)
    Y = (R_target + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)
    
    toroid_positions = np.vstack([X, Y, Z]).T
    
    # Keep atoms within the circumference plus overlap zone
    # This ensures we have overlap to detect and remove duplicates
    seam_mask = s < (L_target + overlap_tolerance)
    toroid_positions = toroid_positions[seam_mask]
    toroid_symbols = cyl_symbols[seam_mask].tolist()
    
    print(f"\n✓ Cylinder mapped to toroid geometry")
    print(f"  Atoms after mapping: {len(toroid_symbols)}")
    print(f"  Major radius R: {R_target:.3f} Å")
    print(f"  Minor radius r: {r_cyl:.3f} Å")
    
    return toroid_symbols, toroid_positions


def remove_duplicate_atoms(symbols, positions, overlap_tolerance):
    """
    Remove duplicate atoms at the toroid seam using neighbor list detection.
    
    This is the most sophisticated approach: it builds a neighbor list
    with the given tolerance and removes atoms that are too close to each other.
    For pairs of duplicates, it keeps the atom with lower index.
    
    Parameters:
        symbols (list): Chemical symbols
        positions (array): Atomic positions
        overlap_tolerance (float): Distance threshold for considering atoms as duplicates
    
    Returns:
        Atoms: ASE Atoms object with duplicates removed
    """
    # Create temporary Atoms object
    temp_atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    
    print(f"\n✓ Removing duplicate atoms at seam")
    print(f"  Atoms before removal: {len(temp_atoms)}")
    print(f"  Overlap tolerance: {overlap_tolerance:.3f} Å")
    
    # Build neighbor list with cutoff based on overlap_tolerance
    # Use overlap_tolerance as cutoff to ensure we find all potential duplicates
    try:
        # Create uniform cutoffs based on overlap_tolerance for all atoms
        cutoffs = [overlap_tolerance] * len(temp_atoms)
        nl = build_neighbor_list(temp_atoms, cutoffs=cutoffs, skin=0.0)
        
        # Find atoms that are too close (duplicates)
        atoms_to_delete = set()
        for i in range(len(temp_atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                # Avoid double-counting and self-comparison
                if i >= j:
                    continue
                
                # Calculate actual distance
                dist = temp_atoms.get_distance(i, j, mic=False)
                
                # Mark for deletion if too close
                if dist < overlap_tolerance:
                    # Keep lower index, delete higher index
                    atoms_to_delete.add(max(i, j))
        
        # Remove marked atoms efficiently
        if atoms_to_delete:
            print(f"  Duplicates found: {len(atoms_to_delete)}")
            # Create new Atoms object with only kept atoms for efficiency
            keep_indices = [i for i in range(len(temp_atoms)) if i not in atoms_to_delete]
            temp_atoms = temp_atoms[keep_indices]
        else:
            print(f"  No duplicates found")
    
    except Exception as e:
        print(f"  Warning: Could not build neighbor list: {e}")
        print(f"  Proceeding without duplicate removal")
    
    print(f"  Atoms after removal: {len(temp_atoms)}")
    
    return temp_atoms
