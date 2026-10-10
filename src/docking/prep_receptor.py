import sys
import numpy as np
import openmm as mm
from openmm import app, unit, Vec3
from pdbfixer import PDBFixer
from openmm.app import PDBFile
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
topology, positions = fixer.topology, fixer.positions

# Short, gently restrained energy minimisation (like ColabFold's Amber relax, which was skipped).
# It removes the tiny atom clashes that unrelaxed models contain and that make Meeko fail.
try:
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff)
    restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 100.0)           # kJ/mol/nm^2, gentle
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    start = np.array(positions.value_in_unit(nanometer))
    heavy = []
    for atom in topology.atoms():
        if atom.element is not None and atom.element.symbol != "H":
            restraint.addParticle(atom.index, [float(v) for v in start[atom.index]])
            heavy.append(atom.index)
    system.addForce(restraint)
    sim = app.Simulation(topology, system, mm.VerletIntegrator(0.001))
    sim.context.setPositions(positions)
    sim.minimizeEnergy(tolerance=10 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=1000)
    positions = sim.context.getState(getPositions=True).getPositions()
    end = np.array(positions.value_in_unit(nanometer))
    shift = 10 * np.linalg.norm(end[heavy] - start[heavy], axis=1)
    print(f"Minimised: heavy-atom movement mean {shift.mean():.2f} A, max {shift.max():.2f} A")
except Exception as e:
    print(f"WARNING: minimisation skipped ({type(e).__name__}: {e}). Output is NOT relaxed.")

with open(outfile, "w") as f:
    PDBFile.writeFile(topology, positions, f)
print("Wrote", outfile, "with", topology.getNumAtoms(), "atoms")