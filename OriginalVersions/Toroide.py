import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar

# ==========================
# PARÁMETROS DEL TOROIDE
# ==========================
R_target = 8.0   # Å  → radio mayor del toroide (centro del agujero al centro del tubo)
r_cyl    = 4.0    # Å  → radio del cilindro previo (grosor del anillo)
seam_pad = 2.0    # Å  → grosor de "banda" que recortamos para evitar duplicados en la costura

# (Opcional) pre-relajación con Lennard-Jones para quitar choques
DO_PRERELAX = False
LJ_epsilon  = 0.002  # eV (muy suave)
LJ_sigma    = 2.5    # Å
PRERELAX_STEPS = 200

# ==========================
# 1) Lee celda unitaria
# ==========================
atoms0 = read("Li2O2.cif")
atoms0.set_pbc([True, True, True])
atoms0.wrap()

# Tomamos a_x ~ longitud del vector de celda en x
a_lengths = cell_to_cellpar(atoms0.get_cell())[:3]
a_x = a_lengths[0]
a_y = a_lengths[1]
a_z = a_lengths[2]

# ==========================
# 2) Elegir longitud del cilindro ~ 2πR_target
#    y número de repeticiones en x para aproximarla
# ==========================
L_target = 2.0 * np.pi * R_target
nx = max(2, int(np.ceil(L_target / a_x)))

# Repeticiones en y,z para cubrir r_cyl
# Queremos que el bloque cubra al menos 2*r_cyl en y y z
ny = max(2, int(np.ceil((2.2 * r_cyl) / a_y)))  # 2.2 factor seguridad
nz = max(2, int(np.ceil((2.2 * r_cyl) / a_z)))

P = np.diag([nx, ny, nz])
block = make_supercell(atoms0, P)
block.set_pbc([False, False, False])  # trabajaremos como clúster
block.wrap()

# Dimensiones del bloque
cell = block.get_cell()
Lx = nx * a_x
Ly = ny * a_y
Lz = nz * a_z

# ==========================
# 3) Centrar el bloque y definir cilindro
#    Eje del cilindro: x
#    Centro transversal: (y0, z0)
# ==========================
pos = block.get_positions()
x_min, x_max = pos[:,0].min(), pos[:,0].max()
y_min, y_max = pos[:,1].min(), pos[:,1].max()
z_min, z_max = pos[:,2].min(), pos[:,2].max()

# Trasladamos para que:
#   x en [0, Lx]
#   y,z centrados en 0
block.translate([-x_min, -(y_min + y_max)/2.0, -(z_min + z_max)/2.0])
pos = block.get_positions()

# Recalcular rangos útiles
L = pos[:,0].max() - pos[:,0].min()  # ~ Lx real
# Filtrar solo los átomos dentro del radio cilíndrico en y-z
y = pos[:,1]
z = pos[:,2]
r_perp = np.sqrt(y**2 + z**2)
keep = r_perp <= r_cyl

cyl_symbols   = np.array(block.get_chemical_symbols())[keep]
cyl_positions = pos[keep]

# ==========================
# 4) Mapear cilindro → toroide
#    s = coordenada a lo largo del eje x ∈ [0, L]
#    theta = 2π * s / L
#    (r_perp, phi) desde (y,z)
#    Ecuaciones:
#      X = (R + r_perp*cos phi) * cos(theta)
#      Y = (R + r_perp*cos phi) * sin(theta)
#      Z =  r_perp * sin(phi)
# ==========================
s = cyl_positions[:,0] - cyl_positions[:,0].min()   # en [0, L]
y = cyl_positions[:,1]
z = cyl_positions[:,2]
phi = np.arctan2(z, y)         # ángulo transversal
r  = np.sqrt(y**2 + z**2)      # radio transversal

theta = 2.0 * np.pi * (s / L)  # cierra perfectamente

X = (R_target + r * np.cos(phi)) * np.cos(theta)
Y = (R_target + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

toroid_positions = np.vstack([X, Y, Z]).T
toroid_symbols   = cyl_symbols.tolist()

# ==========================
# 5) Eliminar duplicados en la "costura"
#    Quitamos una franja de ancho 'seam_pad' cerca de s ~ L
# ==========================
seam_mask = s <= (L - seam_pad)
toroid_positions = toroid_positions[seam_mask]
toroid_symbols   = np.array(toroid_symbols)[seam_mask].tolist()

toroid = Atoms(symbols=toroid_symbols, positions=toroid_positions, pbc=False)

print("Toroide preliminar:")
print("  Átomos:", len(toroid))
print("  R (mayor):", R_target, "Å")
print("  r (menor ~ r_cyl):", r_cyl, "Å")

# ==========================
# 6) (OPCIONAL) Pre-relajación suave
#    Única finalidad: eliminar solapamientos fuertes antes de DFT/MD reales.
#    NO es químicamente fiel para Li2O2 → usar DFT/MD después.
# ==========================
if DO_PRERELAX:
    try:
        from ase.calculators.lj import LennardJones
        from ase.optimize import BFGS
        calc = LennardJones(epsilon=LJ_epsilon, sigma=LJ_sigma, rc=3*LJ_sigma)
        toroid.calc = calc
        dyn = BFGS(toroid, logfile=None)
        dyn.run(fmax=0.1, steps=PRERELAX_STEPS)
        print("Pre-relajación LJ terminada.")
    except Exception as e:
        print("Aviso: no se pudo pre-relajar con LJ:", e)

# ==========================
# 7) Guardar
# ==========================
write("Li2O2_toroide.xyz", toroid)
print("Guardado: Li2O2_toroide.xyz")
