import argparse, csv, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--names", nargs="*", default=[], help="run names; if omitted, every folder in DATA_ROOT that holds a NAME_traj.dcd is analysed")
ap.add_argument("--out-root", default=os.path.join("results", "md"))
ap.add_argument("--frame-ps", default="100")
ap.add_argument("--skip-ns", default="5")
ap.add_argument("--ser", default="132")
ap.add_argument("--asp", default="178")
ap.add_argument("--his", default="209")
args = ap.parse_args()

if not args.names:
    args.names = sorted(d for d in os.listdir(args.data_root)
                        if os.path.exists(os.path.join(args.data_root, d, f"{d}_traj.dcd")))
    print(f"Found {len(args.names)} runs in {args.data_root}: " + ", ".join(args.names))

summaries = {}
for name in args.names:
    d = os.path.join(args.data_root, name)
    start, traj, log = (os.path.join(d, f"{name}{s}") for s in ("_start.pdb", "_traj.dcd", "_log.csv"))
    if not all(os.path.exists(p) for p in (start, traj, log)):
        print(f"{name}: files not found in {d} (need _start.pdb, _traj.dcd, _log.csv) - skipped"); continue
    rows = list(csv.reader(open(log)))[1:]
    temps = [float(r[3]) for r in rows if len(r) > 3]
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "analysis.log"), "w") as lf:
        rc = subprocess.run([sys.executable, os.path.join(HERE, "analyze_md.py"), "--start", start, "--traj", traj,
                             "--name", name, "--out-dir", out_dir, "--frame-ps", args.frame_ps, "--skip-ns", args.skip_ns,
                             "--ser", args.ser, "--asp", args.asp, "--his", args.his], stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"{name}: analysis FAILED - see {os.path.join(out_dir, 'analysis.log')}"); continue
    s = json.load(open(os.path.join(out_dir, f"{name}_summary.json")))
    s["mean_T_K"] = sum(temps) / len(temps)
    summaries[name] = s

if not summaries:
    raise SystemExit("Nothing analysed.")

cols = [("total_ns", "ns", "{:.1f}"), ("mean_T_K", "T(K)", "{:.0f}"), ("rmsd_mean_A", "RMSD", "{:.2f}"),
        ("rmsd_drift_A_per_ns", "drift/ns", "{:+.3f}"), ("rg_mean_A", "Rg", "{:.2f}"), ("sasa_mean_nm2", "SASA", "{:.1f}"),
        ("rmsf_mean_A", "RMSF", "{:.2f}"), ("d_SerOG_HisNE2_mean_A", "S-H", "{:.2f}"),
        ("d_HisND1_AspOD_mean_A", "H-D", "{:.2f}"), ("triad_intact_fraction", "triad", "{:.0%}")]
print(f"\n{'run':<20}" + "".join(f"{c[1]:>10}" for c in cols))
for name, s in summaries.items():
    print(f"{name:<20}" + "".join(f"{c[2].format(s[c[0]]):>10}" for c in cols))

wt = [s for n, s in summaries.items() if n.upper().startswith("WT")]
if len(wt) >= 2:
    print(f"\nWT band over {len(wt)} replicates (min - max):")
    band = {}
    for key, label, fmt in cols[2:]:
        vals = [s[key] for s in wt]
        band[key] = (min(vals), max(vals))
        print(f"  {label:<10}{fmt.format(min(vals))} - {fmt.format(max(vals))}")
    others = {n: s for n, s in summaries.items() if not n.upper().startswith("WT")}
    if others:
        print("\nRuns outside the WT band (above = higher than every WT replicate, below = lower than every one):")
        for n, s in others.items():
            out = []
            for key, label, fmt in cols[2:]:
                if s[key] > band[key][1]: out.append(f"{label} above")
                elif s[key] < band[key][0]: out.append(f"{label} below")
            print(f"  {n:<20}{', '.join(out) if out else 'inside the WT band on every metric'}")
    print("\nNote: with only a few WT replicates the band is crude; treat 'outside' as a hint, not a result.")
else:
    print("\n(Need at least 2 runs whose name starts with 'WT' to form a band.)")