import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar

# ==========================
# PARÁMETROS DEL TOROIDE
# ==========================
R_target = 6.0   # Å - radio mayor
r_cyl    = 3.5   # Å - radio del cilindro previo
seam_pad = 1.0   # Å - ancho de la franja que quitamos en la costura (ajustable)

# (Opcional) pre-relajación con Lennard-Jones
DO_PRERELAX = False
LJ_epsilon  = 0.002
LJ_sigma    = 2.5
PRERELAX_STEPS = 200

# ==========================
# 1) Lee celda unitaria
# ==========================
atoms0 = read("Li2O2.cif")
atoms0.set_pbc([True, True, True])
atoms0.wrap()

# obtener longitudes de celda
a_lengths = cell_to_cellpar(atoms0.get_cell())[:3]
a_x = a_lengths[0]
a_y = a_lengths[1]
a_z = a_lengths[2]

# ==========================
# 2) Elegir nx para aproximar L_target
# ==========================
L_target = 2.0 * np.pi * R_target

# usar round en vez de ceil para acercar L a L_target
nx = max(2, int(np.round(L_target / a_x)))
L_est = nx * a_x
print(f"nx={nx}, L_est={L_est:.6f} Å, L_target={L_target:.6f} Å, diff={L_est - L_target:.6f} Å")

# repeticiones en y,z (cobertura con factor de seguridad)
ny = max(2, int(np.ceil((2.2 * r_cyl) / a_y)))
nz = max(2, int(np.ceil((2.2 * r_cyl) / a_z)))

P = np.diag([nx, ny, nz])
block = make_supercell(atoms0, P)
block.set_pbc([False, False, False])
block.wrap()

# dimensiones del bloque (usamos L = nx * a_x para consistencia)
Lx = nx * a_x
Ly = ny * a_y
Lz = nz * a_z

# ==========================
# 3) Centrar el bloque y definir cilindro
# ==========================
pos = block.get_positions()
x_min, x_max = pos[:,0].min(), pos[:,0].max()
y_min, y_max = pos[:,1].min(), pos[:,1].max()
z_min, z_max = pos[:,2].min(), pos[:,2].max()

# trasladamos: x en [0, Lx], y,z centrados en 0
block.translate([-x_min, -(y_min + y_max)/2.0, -(z_min + z_max)/2.0])
pos = block.get_positions()

# Asegurar que x esté en [0, Lx) (evita valores en los bordes por redondeo)
pos[:,0] = np.mod(pos[:,0], Lx)
block.set_positions(pos)

# Filtrar solo los átomos dentro del radio cilíndrico en y-z
pos = block.get_positions()
y = pos[:,1]
z = pos[:,2]
r_perp = np.sqrt(y**2 + z**2)
keep = r_perp <= r_cyl + 1e-8  # pequeña tolerancia
cyl_symbols   = np.array(block.get_chemical_symbols())[keep]
cyl_positions = pos[keep]

# ==========================
# 4) Normalizar y ordenar por s (coordenada axial)
# ==========================
# s en [0, Lx)
s = cyl_positions[:,0]
# normalizamos por seguridad a [0, Lx)
s = np.mod(s, Lx)

# orden por s para continuidad angular
order = np.argsort(s)
s = s[order]
cyl_positions = cyl_positions[order]
cyl_symbols = cyl_symbols[order]

# ==========================
# 5) Mapear cilindro -> toroide
# ==========================
phi = np.arctan2(cyl_positions[:,2], cyl_positions[:,1])
r   = np.sqrt(cyl_positions[:,1]**2 + cyl_positions[:,2]**2)

theta = 2.0 * np.pi * (s / Lx)   # clave: usar Lx = nx*a_x
X = (R_target + r * np.cos(phi)) * np.cos(theta)
Y = (R_target + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

toroid_positions = np.vstack([X, Y, Z]).T
toroid_symbols   = cyl_symbols.tolist()

# ==========================
# 6) Eliminar duplicados en la "costura"
#    Quitamos átomos con s > Lx - seam_pad
# ==========================
if seam_pad > 0.0:
    mask = s <= (Lx - seam_pad)
    removed = np.count_nonzero(~mask)
    print(f"Seam pad: {seam_pad} Å -> Eliminando {removed} átomos en la franja de costura.")
    toroid_positions = toroid_positions[mask]
    toroid_symbols = np.array(toroid_symbols)[mask].tolist()
else:
    print("Seam pad = 0: no se remueve franja de costura (puede dejar duplicados).")

toroid = Atoms(symbols=toroid_symbols, positions=toroid_positions, pbc=False)

print("Toroide preliminar:")
print("  Átomos:", len(toroid))
print("  R (mayor):", R_target, "Å")
print("  r_cyl (menor ~):", r_cyl, "Å")

# ==========================
# 7) (Opcional) Pre-relajación LJ
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
# 8) Guardar
# ==========================
write("Li2O2_toroide.xyz", toroid)
print("Guardado: Li2O2_toroide.xyz")

