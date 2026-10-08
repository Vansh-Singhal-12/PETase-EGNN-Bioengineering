import sys
import numpy as np
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from openmm import Vec3
from openmm.unit import nanometer, Quantity

infile, outfile = sys.argv[1], sys.argv[2]
fixer = PDBFixer(infile)
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()

# PDBFixer places the C-terminal OXT atom too close to CA, which later breaks Meeko.
# Re-place it properly (flat carboxylate, 1.25 Angstrom from C).
coords = [np.array(p) for p in fixer.positions.value_in_unit(nanometer)]
for res in fixer.topology.residues():
    idx = {a.name: a.index for a in res.atoms()}
    if all(n in idx for n in ("OXT", "CA", "C", "O")):
        c, ca, o = coords[idx["C"]], coords[idx["CA"]], coords[idx["O"]]
        u1 = (ca - c) / np.linalg.norm(ca - c)
        u2 = (o - c) / np.linalg.norm(o - c)
        d = -(u1 + u2)
        d = d / np.linalg.norm(d)
        coords[idx["OXT"]] = c + 0.125 * d
        dist = 10 * np.linalg.norm(coords[idx["OXT"]] - ca)
        print(f"Re-placed OXT on residue {res.id}: OXT-CA distance now {dist:.2f} Angstrom (should be ~2.4)")
fixer.positions = Quantity([Vec3(*x) for x in coords], nanometer)

fixer.addMissingHydrogens(7.0)   # pH 7.0
with open(outfile, "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)
print("Wrote", outfile, "with", fixer.topology.getNumAtoms(), "atoms")