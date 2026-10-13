"""
analyze_md.py - turn one finished (or partly finished) MD run into numbers.

Needs: mdtraj  (conda install -c conda-forge mdtraj   or   pip install mdtraj)

Usage (inside the activated 'dock' environment):
  python analyze_md.py --start C:\md_data\WT_rep1\WT_rep1_start.pdb --traj C:\md_data\WT_rep1\WT_rep1_traj.dcd ^
         --name WT_rep1 --out-dir results\md\WT_rep1

What it measures (all in Angstrom unless stated; analysis window = after --skip-ns):
  RMSD      backbone RMSD to the start structure (frames superposed on backbone, termini excluded)
  RMSF      per-residue C-alpha fluctuation about the average structure (saved per residue, real numbering)
  Rg        radius of gyration of protein heavy atoms
  SASA      solvent-accessible surface area (nm^2)
  Triad     Ser160 OG - His237 NE2 and His237 ND1 - Asp206 OD1/OD2 distances; 'triad intact' = both <= 3.5 A
Files written: NAME_timeseries.csv, NAME_rmsf.csv, NAME_summary.json
"""
import argparse, csv, json, os
import numpy as np
import mdtraj as md

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True, help="NAME_start.pdb written by run_md.py (topology)")
ap.add_argument("--traj", required=True, help="NAME_traj.dcd")
ap.add_argument("--name", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--frame-ps", type=float, default=100.0, help="time between saved frames (run_md.py default 100 ps)")
ap.add_argument("--skip-ns", type=float, default=5.0, help="discard this much from the start before averaging")
ap.add_argument("--ser", type=int, default=132, help="file residue number of Ser160")
ap.add_argument("--asp", type=int, default=178, help="file residue number of Asp206")
ap.add_argument("--his", type=int, default=209, help="file residue number of His237")
ap.add_argument("--offset", type=int, default=28, help="real position = file residue + offset")
ap.add_argument("--sasa-stride", type=int, default=5)
ap.add_argument("--cutoff", type=float, default=3.5, help="H-bond style cutoff for 'triad intact' (Angstrom)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

ref = md.load(args.start)
traj = md.load(args.traj, top=ref.topology)
n = traj.n_frames
times_ns = (np.arange(n) + 1) * args.frame_ps / 1000.0
keep = times_ns > args.skip_ns
print(f"{args.name}: {n} frames, {times_ns[-1]:.1f} ns total; analysing {int(keep.sum())} frames after {args.skip_ns} ns")
if keep.sum() < 3:
    raise SystemExit("Too few frames after --skip-ns; lower it.")

top = ref.topology
res_nums = [r.resSeq for r in top.residues if r.is_protein]
lo, hi = min(res_nums) + 2, max(res_nums) - 2                       # leave out 2 floppy residues at each end
core_bb = top.select(f"protein and backbone and resSeq {lo} to {hi}")
core_ca = top.select(f"protein and name CA and resSeq {lo} to {hi}")
all_ca = top.select("protein and name CA")
heavy = top.select("protein and not element H")

rmsd = md.rmsd(traj, ref, atom_indices=core_bb) * 10.0
rg = md.compute_rg(traj.atom_slice(heavy)) * 10.0
sasa = np.full(n, np.nan)
idx = np.arange(0, n, args.sasa_stride)
sasa[idx] = md.shrake_rupley(traj[idx]).sum(axis=1)

def atom(resnum, names):
    sel = top.select(f"resSeq {resnum} and (" + " or ".join(f"name {a}" for a in names) + ")")
    if len(sel) != len(names):
        raise SystemExit(f"Could not find atoms {names} in residue {resnum}; check --ser/--his/--asp")
    return sel
og, ne2, nd1 = atom(args.ser, ["OG"])[0], atom(args.his, ["NE2"])[0], atom(args.his, ["ND1"])[0]
od = atom(args.asp, ["OD1", "OD2"])
d_sh = md.compute_distances(traj, [[og, ne2]])[:, 0] * 10.0
d_hd = np.min(md.compute_distances(traj, [[nd1, od[0]], [nd1, od[1]]]), axis=1) * 10.0
intact = (d_sh <= args.cutoff) & (d_hd <= args.cutoff)

# RMSF about the average structure after fitting on the core C-alpha atoms
fit = traj[keep]
fit.superpose(fit, 0, atom_indices=core_ca)
xyz = fit.xyz[:, all_ca, :]
rmsf = np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0)) * 10.0
resid = [top.atom(i).residue.resSeq for i in all_ca]

with open(os.path.join(args.out_dir, f"{args.name}_timeseries.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time_ns", "rmsd_A", "rg_A", "sasa_nm2", "d_SerOG_HisNE2_A", "d_HisND1_AspOD_A"])
    for i in range(n):
        w.writerow([round(times_ns[i], 3), round(rmsd[i], 3), round(rg[i], 3),
                    "" if np.isnan(sasa[i]) else round(sasa[i], 3), round(d_sh[i], 3), round(d_hd[i], 3)])
with open(os.path.join(args.out_dir, f"{args.name}_rmsf.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file_residue", "real_position", "rmsf_A"])
    for r, v in zip(resid, rmsf):
        w.writerow([r, r + args.offset, round(float(v), 3)])

t = times_ns[keep]
slope = float(np.polyfit(t, rmsd[keep], 1)[0])
summary = {
    "name": args.name, "frames": int(n), "total_ns": round(float(times_ns[-1]), 2), "window_ns": [args.skip_ns, round(float(times_ns[-1]), 2)],
    "rmsd_mean_A": round(float(rmsd[keep].mean()), 3), "rmsd_sd_A": round(float(rmsd[keep].std()), 3),
    "rmsd_max_A": round(float(rmsd[keep].max()), 3), "rmsd_drift_A_per_ns": round(slope, 4),
    "rg_mean_A": round(float(rg[keep].mean()), 3), "rg_sd_A": round(float(rg[keep].std()), 3),
    "sasa_mean_nm2": round(float(np.nanmean(sasa[keep])), 2),
    "rmsf_mean_A": round(float(rmsf.mean()), 3), "rmsf_max_A": round(float(rmsf.max()), 3),
    "rmsf_max_real_position": int(resid[int(rmsf.argmax())] + args.offset),
    "d_SerOG_HisNE2_mean_A": round(float(d_sh[keep].mean()), 3), "d_SerOG_HisNE2_sd_A": round(float(d_sh[keep].std()), 3),
    "d_HisND1_AspOD_mean_A": round(float(d_hd[keep].mean()), 3), "d_HisND1_AspOD_sd_A": round(float(d_hd[keep].std()), 3),
    "triad_intact_fraction": round(float(intact[keep].mean()), 3), "triad_cutoff_A": args.cutoff,
}
with open(os.path.join(args.out_dir, f"{args.name}_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'RMSD (backbone)':<26}{summary['rmsd_mean_A']:.2f} +/- {summary['rmsd_sd_A']:.2f} A   (max {summary['rmsd_max_A']:.2f}; drift {slope:+.3f} A/ns)")
print(f"{'Radius of gyration':<26}{summary['rg_mean_A']:.2f} +/- {summary['rg_sd_A']:.2f} A")
print(f"{'SASA':<26}{summary['sasa_mean_nm2']:.1f} nm^2")
print(f"{'RMSF (mean / max)':<26}{summary['rmsf_mean_A']:.2f} / {summary['rmsf_max_A']:.2f} A (max at real position {summary['rmsf_max_real_position']})")
print(f"{'Ser OG - His NE2':<26}{summary['d_SerOG_HisNE2_mean_A']:.2f} +/- {summary['d_SerOG_HisNE2_sd_A']:.2f} A")
print(f"{'His ND1 - Asp OD':<26}{summary['d_HisND1_AspOD_mean_A']:.2f} +/- {summary['d_HisND1_AspOD_sd_A']:.2f} A")
print(f"{'Triad intact (<= %.1f A)' % args.cutoff:<26}{summary['triad_intact_fraction']:.0%} of frames")
print(f"\nWrote {args.name}_timeseries.csv, {args.name}_rmsf.csv, {args.name}_summary.json in {args.out_dir}")
