import argparse, os, sys, time
import numpy as np
import openmm as mm
from openmm import app, unit

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True, help="prepared structure with hydrogens, e.g. candidate_001_H.pdb")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--ns", type=float, required=True, help="TOTAL production length in ns")
ap.add_argument("--temp", type=float, default=338.0, help="Kelvin (338 K = 65 C)")
ap.add_argument("--dt-fs", type=float, default=4.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--report-ps", type=float, default=100.0, help="trajectory frame spacing")
ap.add_argument("--chunk-ns", type=float, default=1.0, help="checkpoint every this many ns")
ap.add_argument("--equil-scale", type=float, default=1.0, help="testing only: shrink equilibration")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
P = lambda suffix: os.path.join(args.out_dir, args.name + suffix)
dt_ps = args.dt_fs / 1000.0
steps_per_ns = int(round(1000.0 / dt_ps))
total_steps = int(round(args.ns * steps_per_ns))

# ---------------- system ----------------
pdb = app.PDBFile(args.pdb)
ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                         constraints=app.HBonds, hydrogenMass=1.5 * unit.amu)
restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
restraint.addGlobalParameter("k", 0.0)
for n in ("x0", "y0", "z0"):
    restraint.addPerParticleParameter(n)
start_xyz = np.array(pdb.positions.value_in_unit(unit.nanometer))
for a in pdb.topology.atoms():
    if a.element is not None and a.element.symbol != "H":
        restraint.addParticle(a.index, [float(v) for v in start_xyz[a.index]])
system.addForce(restraint)

integrator = mm.LangevinMiddleIntegrator(50 * unit.kelvin, 1.0 / unit.picosecond, dt_ps * unit.picosecond)
integrator.setRandomNumberSeed(args.seed)

names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
pname = next(n for n in ("CUDA", "HIP", "OpenCL", "CPU") if n in names)
props = {"Precision": "mixed"} if pname in ("CUDA", "OpenCL") else {}
sim = app.Simulation(pdb.topology, system, integrator, mm.Platform.getPlatformByName(pname), props)
print(f"Platform: {pname} | OpenMM {mm.version.version} | {pdb.topology.getNumAtoms()} atoms | "
      f"{args.temp} K | dt {args.dt_fs} fs | target {args.ns} ns")

def set_k(v):
    sim.context.setParameter("k", v)

def run_steps(n):
    sim.step(int(max(1, round(n * args.equil_scale))))

chk, resuming = P(".chk"), os.path.exists(P(".chk"))
if resuming:
    sim.loadCheckpoint(chk)
    print(f"Resuming from checkpoint at {sim.context.getState().getTime().value_in_unit(unit.nanosecond):.3f} ns")
else:
    with open(P("_settings.txt"), "w") as f:
        f.write(" ".join(sys.argv) + f"\nplatform={pname} openmm={mm.version.version}\n")
    sim.context.setPositions(pdb.positions)
    set_k(1000.0)                                   # kJ/mol/nm^2 on heavy atoms
    sim.minimizeEnergy(maxIterations=1000)
    sim.context.setVelocitiesToTemperature(50 * unit.kelvin, args.seed)
    print("Heating (restrained)...")
    for T in np.linspace(50, args.temp, 8):
        integrator.setTemperature(float(T) * unit.kelvin)
        run_steps(2500)                             # 10 ps per stage
    print("Equilibrating (restraints released step by step)...")
    for k in (1000.0, 100.0, 10.0):
        set_k(k)
        run_steps(25000)                            # 100 ps each
    set_k(0.0)
    run_steps(25000)                                # 100 ps unrestrained
    state = sim.context.getState(getPositions=True)
    with open(P("_start.pdb"), "w") as f:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
    sim.context.setTime(0.0)
    try:
        sim.context.setStepCount(0)
    except AttributeError:
        pass
    sim.saveCheckpoint(chk)
    print("Equilibration done; production starts.")

# the thermostat temperature is not stored in the checkpoint, so always set it before production
integrator.setTemperature(args.temp * unit.kelvin)
print(f"Thermostat set to {integrator.getTemperature().value_in_unit(unit.kelvin):.0f} K for production")

report_steps = int(round(args.report_ps / dt_ps))
sim.reporters.append(app.DCDReporter(P("_traj.dcd"), report_steps, append=resuming))
sim.reporters.append(app.StateDataReporter(P("_log.csv"), report_steps, step=True, time=True,
                     potentialEnergy=True, temperature=True, append=resuming))

done = int(round(sim.context.getState().getTime().value_in_unit(unit.picosecond) / dt_ps))
remaining = total_steps - done
if remaining <= 0:
    print(f"Already at {done / steps_per_ns:.3f} ns (target {args.ns} ns). Nothing to do; raise --ns to extend.")
else:
    chunk = int(round(args.chunk_ns * steps_per_ns))
    t0 = time.time()
    ran = 0
    while remaining > 0:
        n = min(chunk, remaining)
        sim.step(n)
        remaining -= n; ran += n
        sim.saveCheckpoint(chk + ".tmp"); os.replace(chk + ".tmp", chk)       # safe overwrite
        sim_ns = (total_steps - remaining) / steps_per_ns
        speed = (ran / steps_per_ns) / ((time.time() - t0) / 86400.0)
        print(f"  {sim_ns:8.3f} / {args.ns} ns   speed {speed:7.1f} ns/day", flush=True)

state = sim.context.getState(getPositions=True)
with open(P("_final.pdb"), "w") as f:
    app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
print("Finished. Files are in", args.out_dir)
