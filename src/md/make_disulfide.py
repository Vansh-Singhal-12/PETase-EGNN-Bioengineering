"""
make_disulfide.py - build real disulfide bonds into a prepared structure, BEFORE running MD.

Why: ColabFold did not know an engineered Cys pair should bond, so in some models the two sulfur atoms sit
~4.5 A apart and carry thiol hydrogens. Run as-is, the MD would simulate a REDUCED (unbonded) pair. This script
(1) rotates each Cys side chain to the best position for forming the S-S bond, (2) deletes the thiol
hydrogens, (3) registers the bond, and (4) does a gentle energy minimisation so the geometry relaxes.

Usage (inside the activated 'dock' environment); positions are REAL numbering (file residue = real - 28):
  python make_disulfide.py --pdb C:\docking\runs\candidate_062\candidate_062_H.pdb --out C:\PETase_md_data\inputs\candidate_062_SS.pdb --pair 233-282
  (repeat --pair for several pairs; pairs that are already bonded are left alone)
"""
import argparse
import numpy as np
import openmm as mm
from openmm import app, unit, Vec3

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pair", action="append", required=True, help="real positions, e.g. 233-282")
ap.add_argument("--offset", type=int, default=28, help="file residue = real position - offset")
ap.add_argument("--max-gap", type=float, default=1.8, help="give up if the best S-S distance after side-chain rotation is this far (A) from 2.05")
args = ap.parse_args()

pdb = app.PDBFile(args.pdb)
top = pdb.topology
pos = np.array(pdb.positions.value_in_unit(unit.angstrom))
resmap = {int(r.id): r for r in top.residues()}
idx = lambda res, name: next((a.index for a in res.atoms() if a.name == name), None)

def rotate(points, origin, axis, theta):
    k = axis / np.linalg.norm(axis)
    p = points - origin
    return origin + p * np.cos(theta) + np.cross(k, p) * np.sin(theta) + np.outer(p @ k, k) * (1 - np.cos(theta))

heavy = np.array([a.element is not None and a.element.symbol != "H" for a in top.atoms()])
angles = np.deg2rad(np.arange(0, 360, 5))
todo = []
for p in args.pair:
    a, b = (int(x) for x in p.split("-"))
    ra, rb = resmap[a - args.offset], resmap[b - args.offset]
    for r, real in ((ra, a), (rb, b)):
        if r.name not in ("CYS", "CYX") or idx(r, "SG") is None:
            raise SystemExit(f"Residue at real position {real} is {r.name}, not a cysteine")
    sa, sb = idx(ra, "SG"), idx(rb, "SG")
    d0 = np.linalg.norm(pos[sa] - pos[sb])
    if d0 < 2.3 and idx(ra, "HG") is None and idx(rb, "HG") is None:
        print(f"{a}-{b}: already bonded (S-S {d0:.2f} A) - left alone"); continue

    sides = {}
    for r in (ra, rb):
        ca, cb = idx(r, "CA"), idx(r, "CB")
        mov = [x.index for x in r.atoms() if x.name in ("SG", "HG", "HB1", "HB2", "HB3")]
        sides[r.index] = (ca, cb, mov)
    other = {ra.index: rb, rb.index: ra}

    def candidates(r):
        ca, cb, mov = sides[r.index]
        axis, origin = pos[cb] - pos[ca], pos[cb]
        sg_i = idx(r, "SG")
        sgs = np.array([rotate(pos[sg_i][None, :], origin, axis, t)[0] for t in angles])
        skip = {x.index for x in r.atoms()} | {idx(other[r.index], "SG"), idx(other[r.index], "CB")}
        mask = heavy.copy(); mask[list(skip)] = False
        env = pos[mask]
        pen = np.array([np.maximum(0, 3.0 - np.linalg.norm(env - s, axis=1)).sum() for s in sgs])
        return sgs, pen

    sga, pena = candidates(ra); sgb, penb = candidates(rb)
    dmat = np.linalg.norm(sga[:, None, :] - sgb[None, :, :], axis=2)
    cost = 4 * np.abs(dmat - 2.05) + pena[:, None] + penb[None, :]
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    print(f"{a}-{b}: S-S {d0:.2f} A -> {dmat[i, j]:.2f} A after rotating the side chains "
          f"(clash penalty {pena[i] + penb[j]:.2f})")
    if abs(dmat[i, j] - 2.05) > args.max_gap:
        raise SystemExit(f"{a}-{b}: the sulfur atoms cannot get close enough (best {dmat[i, j]:.2f} A). "
                         f"This pair is geometrically too far apart for a disulfide in this model.")
    if abs(dmat[i, j] - 2.05) > 0.9:
        print(f"   note: the backbone must shift a little (up to ~{abs(dmat[i, j] - 2.05):.1f} A) to close this bond - check the result")
    for r, t in ((ra, angles[i]), (rb, angles[j])):
        ca, cb, mov = sides[r.index]
        pos[mov] = rotate(pos[mov], pos[cb], pos[cb] - pos[ca], t)
    todo.append((ra, rb))

if not todo:
    raise SystemExit("Nothing to do.")

# --- register the bonds: delete thiol hydrogens, add S-S bond, rename to CYX ---
vec = lambda arr: unit.Quantity([Vec3(*(x / 10.0)) for x in arr], unit.nanometer)
mod = app.Modeller(top, vec(pos))
nums = {int(r.id) for pair in todo for r in pair}
mod.delete([a for a in mod.topology.atoms() if a.name == "HG" and int(a.residue.id) in nums])
newres = {int(r.id): r for r in mod.topology.residues()}
for ra, rb in todo:
    A, B = newres[int(ra.id)], newres[int(rb.id)]
    sa = next(a for a in A.atoms() if a.name == "SG"); sb = next(a for a in B.atoms() if a.name == "SG")
    mod.topology.addBond(sa, sb)
    A.name = B.name = "CYX"

ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff)
tether = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
tether.addGlobalParameter("k", 100.0)
for n in ("x0", "y0", "z0"):
    tether.addPerParticleParameter(n)
start = np.array(mod.positions.value_in_unit(unit.nanometer))
for a in mod.topology.atoms():
    if a.element is not None and a.element.symbol != "H":
        tether.addParticle(a.index, [float(v) for v in start[a.index]])
system.addForce(tether)
sim = app.Simulation(mod.topology, system, mm.VerletIntegrator(0.001))
sim.context.setPositions(mod.positions)
sim.minimizeEnergy(tolerance=10 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=2000)
final = sim.context.getState(getPositions=True).getPositions()
fin = np.array(final.value_in_unit(unit.angstrom))
moved = 10 * np.linalg.norm(np.array(final.value_in_unit(unit.nanometer)) - start, axis=1)
hv = [a.index for a in mod.topology.atoms() if a.element is not None and a.element.symbol != "H"]
print(f"Heavy-atom movement during the final relaxation: mean {moved[hv].mean():.2f} A, max {moved[hv].max():.2f} A")
for ra, rb in todo:
    A, B = newres[int(ra.id)], newres[int(rb.id)]
    sa = next(a.index for a in A.atoms() if a.name == "SG"); sb = next(a.index for a in B.atoms() if a.name == "SG")
    print(f"Final S-S for real {int(ra.id) + args.offset}-{int(rb.id) + args.offset}: {np.linalg.norm(fin[sa] - fin[sb]):.2f} A")
with open(args.out, "w") as f:
    app.PDBFile.writeFile(mod.topology, final, f)
print("Wrote", args.out)
