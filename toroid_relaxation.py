"""
Pre-relaxation functionality for toroid structures.

This module provides Lennard-Jones based pre-relaxation to remove
atomic clashes before more accurate DFT or MD simulations.

Author: Consolidated from multiple versions
Date: 2026-01-23
"""


def prerelax_structure(atoms, epsilon, sigma, max_steps):
    """
    Pre-relax the structure using Lennard-Jones potential.
    
    This is a gentle relaxation to remove strong overlaps before
    more accurate DFT or MD simulations. It is NOT chemically accurate
    for the specific material.
    
    Parameters:
        atoms (Atoms): Structure to relax
        epsilon (float): LJ epsilon parameter in eV
        sigma (float): LJ sigma parameter in Å
        max_steps (int): Maximum optimization steps
    
    Returns:
        Atoms: Relaxed structure
    """
    print(f"\n✓ Pre-relaxing structure with Lennard-Jones potential")
    print(f"  LJ parameters: ε={epsilon} eV, σ={sigma} Å")
    print(f"  Max steps: {max_steps}")
    
    try:
        from ase.calculators.lj import LennardJones
        from ase.optimize import BFGS
        
        # Set up calculator with conservative cutoff radius
        calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=2.5*sigma)
        atoms.calc = calc
        
        # Run optimization
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=0.1, steps=max_steps)
        
        print(f"  Pre-relaxation completed")
    
    except Exception as e:
        print(f"  Warning: Pre-relaxation failed: {e}")
        print(f"  Continuing with un-relaxed structure")
    
    return atoms
