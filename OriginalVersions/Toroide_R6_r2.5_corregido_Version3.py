import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar
from ase.neighborlist import build_neighbor_list, natural_cutoffs # Importamos herramientas para buscar vecinos

# ==========================
# PARÁMETROS DEL TOROIDE
# ==========================
R_target = 5.0   # Å  → radio mayor del toroide (centro del agujero al centro del tubo)
r_cyl    = 3.0    # Å  → radio del cilindro previo (grosor del anillo)

# MODIFICACIÓN: Parámetro para definir la superposición y la tolerancia para eliminar duplicados.
# Un valor típico es la mitad de la distancia Li-O más corta.
# Esto es crucial para cerrar el hueco.
OVERLAP_TOLERANCE = 1.0 # Å

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

a_lengths = cell_to_cellpar(atoms0.get_cell())[:3]
a_x, a_y, a_z = a_lengths[0], a_lengths[1], a_lengths[2]

# ==========================
# 2) Elegir longitud del cilindro
# ==========================
L_target = 2.0 * np.pi * R_target
# MODIFICACIÓN: Nos aseguramos de que el bloque sea más largo que L_target + la zona de solapamiento
nx = max(2, int(np.ceil((L_target + OVERLAP_TOLERANCE) / a_x)))

ny = max(2, int(np.ceil((2.2 * r_cyl) / a_y)))
nz = max(2, int(np.ceil((2.2 * r_cyl) / a_z)))

P = np.diag([nx, ny, nz])
block = make_supercell(atoms0, P)
block.set_pbc([False, False, False])
block.wrap()

# ==========================
# 3) Centrar el bloque y definir cilindro
# ==========================
pos = block.get_positions()
x_min, x_max = pos[:,0].min(), pos[:,0].max()
y_min, y_max = pos[:,1].min(), pos[:,1].max()
z_min, z_max = pos[:,2].min(), pos[:,2].max()

block.translate([-x_min, -(y_min + y_max)/2.0, -(z_min + z_max)/2.0])
pos = block.get_positions()

y_cyl, z_cyl = pos[:,1], pos[:,2]
r_perp = np.sqrt(y_cyl**2 + z_cyl**2)
keep = r_perp <= r_cyl

cyl_symbols   = np.array(block.get_chemical_symbols())[keep]
cyl_positions = pos[keep]

# ==========================
# 4) Mapear cilindro → toroide
# ==========================
s = cyl_positions[:,0] - cyl_positions[:,0].min()
y = cyl_positions[:,1]
z = cyl_positions[:,2]
phi = np.arctan2(z, y)
r  = np.sqrt(y**2 + z**2)

# El mapeo sigue usando L_target para la perfección geométrica
L = L_target
theta = 2.0 * np.pi * (s / L)

X = (R_target + r * np.cos(phi)) * np.cos(theta)
Y = (R_target + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

toroid_positions = np.vstack([X, Y, Z]).T
toroid_symbols   = cyl_symbols.tolist()

# MODIFICACIÓN: Creamos el toroide CON superposición, eliminando solo los átomos
# que están muy lejos del final de la circunferencia.
seam_mask = s < (L_target + OVERLAP_TOLERANCE)
toroid_positions = toroid_positions[seam_mask]
toroid_symbols   = np.array(toroid_symbols)[seam_mask].tolist()

toroid_overlapped = Atoms(symbols=toroid_symbols, positions=toroid_positions, pbc=False)

print(f"Toroide con superposición: {len(toroid_overlapped)} átomos")

# ==========================
# 5) NUEVO: Eliminar duplicados en la costura
# ==========================
# Construimos una lista de vecinos. 'skin=0' significa que usamos la tolerancia exacta.
nl = build_neighbor_list(toroid_overlapped, cutoffs=natural_cutoffs(toroid_overlapped, mult=0.5), skin=0.0)

# Identificamos los átomos que tienen vecinos demasiado cerca (posibles duplicados)
# Un átomo en la costura tendrá un vecino a una distancia muy pequeña.
atoms_to_delete = set()
for i in range(len(toroid_overlapped)):
    indices, offsets = nl.get_neighbors(i)
    for j in indices:
        # Para evitar contar dos veces y no eliminarnos a nosotros mismos
        if i >= j:
            continue
        
        dist = toroid_overlapped.get_distance(i, j, mic=False)
        if dist < OVERLAP_TOLERANCE:
            # Nos quedamos con el átomo de índice menor y marcamos el mayor para eliminarlo
            atoms_to_delete.add(max(i, j))

if atoms_to_delete:
    print(f"Eliminando {len(atoms_to_delete)} átomos duplicados en la costura...")
    # Creamos el toroide final eliminando los átomos marcados
    del toroid_overlapped[sorted(list(atoms_to_delete), reverse=True)]

toroid = toroid_overlapped # Renombramos para que el resto del script funcione

print("Toroide final:")
print("  Átomos:", len(toroid))
print("  R (mayor):", R_target, "Å")
print("  r (menor ~ r_cyl):", r_cyl, "Å")


# ==========================
# 6) (OPCIONAL) Pre-relajación suave
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
write("Li2O2_toroideCorr.xyz", toroid)
print("Guardado: Li2O2_toroideCorr.xyz")
