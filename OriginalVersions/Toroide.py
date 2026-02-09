import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar

# ==========================
# TOROID PARAMETERS
# ==========================
R_target = 8.0   # Å  → major radius of the toroid (center of hole to center of tube)
r_cyl    = 4.0    # Å  → radius of the previous cylinder (ring thickness)
seam_pad = 2.0    # Å  → thickness of "band" we cut to avoid duplicates at the seam

# (Optional) pre-relaxation with Lennard-Jones to remove collisions
DO_PRERELAX = False
LJ_epsilon  = 0.002  # eV (very soft)
LJ_sigma    = 2.5    # Å
PRERELAX_STEPS = 200

# ==========================
# 1) Read unit cell
# ==========================
atoms0 = read("Li2O2.cif")
atoms0.set_pbc([True, True, True])
atoms0.wrap()

# Take a_x ~ length of cell vector in x
a_lengths = cell_to_cellpar(atoms0.get_cell())[:3]
a_x = a_lengths[0]
a_y = a_lengths[1]
a_z = a_lengths[2]

# ==========================
# 2) Choose cylinder length ~ 2πR_target
#    and number of repetitions in x to approximate it
# ==========================
L_target = 2.0 * np.pi * R_target
nx = max(2, int(np.ceil(L_target / a_x)))

# Repetitions in y,z to cover r_cyl
# We want the block to cover at least 2*r_cyl in y and z
ny = max(2, int(np.ceil((2.2 * r_cyl) / a_y)))  # 2.2 safety factor
nz = max(2, int(np.ceil((2.2 * r_cyl) / a_z)))

P = np.diag([nx, ny, nz])
block = make_supercell(atoms0, P)
block.set_pbc([False, False, False])  # we'll work as a cluster
block.wrap()

# Block dimensions
cell = block.get_cell()
Lx = nx * a_x
Ly = ny * a_y
Lz = nz * a_z

# ==========================
# 3) Center the block and define cylinder
#    Cylinder axis: x
#    Transverse center: (y0, z0)
# ==========================
pos = block.get_positions()
x_min, x_max = pos[:,0].min(), pos[:,0].max()
y_min, y_max = pos[:,1].min(), pos[:,1].max()
z_min, z_max = pos[:,2].min(), pos[:,2].max()

# Translate so that:
#   x in [0, Lx]
#   y,z centered at 0
block.translate([-x_min, -(y_min + y_max)/2.0, -(z_min + z_max)/2.0])
pos = block.get_positions()

# Recalculate useful ranges
L = pos[:,0].max() - pos[:,0].min()  # ~ actual Lx
# Filter only atoms within cylindrical radius in y-z
y = pos[:,1]
z = pos[:,2]
r_perp = np.sqrt(y**2 + z**2)
keep = r_perp <= r_cyl

cyl_symbols   = np.array(block.get_chemical_symbols())[keep]
cyl_positions = pos[keep]

# ==========================
# 4) Map cylinder → toroid
#    s = coordinate along x axis ∈ [0, L]
#    theta = 2π * s / L
#    (r_perp, phi) from (y,z)
#    Equations:
#      X = (R + r_perp*cos phi) * cos(theta)
#      Y = (R + r_perp*cos phi) * sin(theta)
#      Z =  r_perp * sin(phi)
# ==========================
s = cyl_positions[:,0] - cyl_positions[:,0].min()   # in [0, L]
y = cyl_positions[:,1]
z = cyl_positions[:,2]
phi = np.arctan2(z, y)         # transverse angle
r  = np.sqrt(y**2 + z**2)      # transverse radius

theta = 2.0 * np.pi * (s / L)  # closes perfectly

X = (R_target + r * np.cos(phi)) * np.cos(theta)
Y = (R_target + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

toroid_positions = np.vstack([X, Y, Z]).T
toroid_symbols   = cyl_symbols.tolist()

# ==========================
# 5) Remove duplicates at the "seam"
#    Remove a strip of width 'seam_pad' near s ~ L
# ==========================
seam_mask = s <= (L - seam_pad)
toroid_positions = toroid_positions[seam_mask]
toroid_symbols   = np.array(toroid_symbols)[seam_mask].tolist()

toroid = Atoms(symbols=toroid_symbols, positions=toroid_positions, pbc=False)

print("Preliminary toroid:")
print("  Atoms:", len(toroid))
print("  R (major):", R_target, "Å")
print("  r (minor ~ r_cyl):", r_cyl, "Å")

# ==========================
# 6) (OPTIONAL) Soft pre-relaxation
#    Only purpose: eliminate strong overlaps before real DFT/MD.
#    NOT chemically faithful for Li2O2 → use DFT/MD afterwards.
# ==========================
if DO_PRERELAX:
    try:
        from ase.calculators.lj import LennardJones
        from ase.optimize import BFGS
        calc = LennardJones(epsilon=LJ_epsilon, sigma=LJ_sigma, rc=3*LJ_sigma)
        toroid.calc = calc
        dyn = BFGS(toroid, logfile=None)
        dyn.run(fmax=0.1, steps=PRERELAX_STEPS)
        print("LJ pre-relaxation finished.")
    except Exception as e:
        print("Warning: could not pre-relax with LJ:", e)

# ==========================
# 7) Save
# ==========================
write("Li2O2_toroide.xyz", toroid)
print("Saved: Li2O2_toroide.xyz")