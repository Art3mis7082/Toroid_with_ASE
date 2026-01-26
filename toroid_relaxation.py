"""Pre-relaxation functionality for toroid structures."""


def prerelax_structure(atoms, epsilon, sigma, max_steps):
    """Pre-relax structure using Lennard-Jones potential."""
    print(f"\n✓ Pre-relaxing structure with Lennard-Jones potential")
    print(f"  LJ parameters: ε={epsilon} eV, σ={sigma} Å")
    print(f"  Max steps: {max_steps}")
    
    try:
        from ase.calculators.lj import LennardJones
        from ase.optimize import BFGS
        
        calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=2.5*sigma)
        atoms.calc = calc
        
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=0.1, steps=max_steps)
        
        print(f"  Pre-relaxation completed")
    
    except Exception as e:
        print(f"  Warning: Pre-relaxation failed: {e}")
        print(f"  Continuing with un-relaxed structure")
    
    return atoms
