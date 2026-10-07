import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

AA = set("ACDEFGHIKLMNPQRSTVWY")
TRIAD = {160: "S", 206: "D", 237: "H"}  # Ser160 / Asp206 / His237 (6EQE numbering)
# FAST-PETase (Lu et al. 2022): S121E / D186H / R224Q / N233K / R280A
FAST_PETASE = [("S", 121, "E"), ("D", 186, "H"), ("R", 224, "Q"),
               ("N", 233, "K"), ("R", 280, "A")]


def read_fasta(path):
    header, seq, n = None, [], 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            n += 1
            if n > 1:
                raise ValueError("more than one FASTA record")
            header = line[1:]
        else:
            seq.append(line)
    if header is None:
        raise ValueError("no FASTA header line")
    return header, "".join(seq).upper().replace(" ", "")


def split_list(s):
    return [x for x in re.split(r"[;,\s]+", (s or "").strip()) if x]


def apply_mutations(wt, muts, offset):
    """muts = [(wt_res, real_pos, mut_res)]. Returns (sequence, [errors])."""
    seq, errs = list(wt), []
    for wt_res, pos, mut_res in muts:
        i = pos - offset
        if not (0 <= i < len(seq)):
            errs.append(f"position {pos} -> index {i} is outside the sequence")
            continue
        if wt[i] != wt_res:
            errs.append(f"{wt_res}{pos}{mut_res}: WT has '{wt[i]}' at index {i}, "
                        f"manifest says '{wt_res}'")
        seq[i] = mut_res
    return "".join(seq), errs


def triad_errors(wt, offset):
    errs = []
    for pos, res in TRIAD.items():
        i = pos - offset
        got = wt[i] if 0 <= i < len(wt) else "out-of-range"
        if got != res:
            errs.append(f"triad residue {res}{pos}: WT has '{got}' at index {i}")
    return errs


def offset_scan(wt, all_muts, base):
    print("\nOffset scan (which offset makes the WT residues line up?):")
    print(f"  {'offset':>6} {'mut mismatches':>16} {'triad mismatches':>18}")
    for o in range(base - 3, base + 4):
        bad = total = 0
        for muts in all_muts:
            for wt_res, pos, _ in muts:
                i = pos - o
                total += 1
                if not (0 <= i < len(wt)) or wt[i] != wt_res:
                    bad += 1
        tbad = len(triad_errors(wt, o))
        flag = "  <-- all consistent" if bad == 0 and tbad == 0 else ""
        print(f"  {o:>6} {f'{bad}/{total}':>16} {tbad:>18}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--offset", type=int, default=29,
                    help="sequence_index = real_position - offset (0-based index)")
    ap.add_argument("--out-dir", default="data/colabfold_inputs")
    ap.add_argument("--force", action="store_true",
                    help="write outputs even if checks fail")
    args = ap.parse_args()

    problems = defaultdict(list)  # candidate_id (or 'GLOBAL') -> [messages]

    _, wt = read_fasta(args.wt)
    print(f"WT sequence: {len(wt)} residues")
    bad_chars = set(wt) - AA
    if bad_chars:
        problems["GLOBAL"].append(f"WT has non-standard characters: {bad_chars}")

    for e in triad_errors(wt, args.offset):
        problems["GLOBAL"].append(e)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"candidate_id", "fasta_file", "wild_type", "mutation_type",
            "position_idx", "n_mutations"}
    missing = need - set(rows[0].keys() if rows else [])
    if missing:
        sys.exit(f"Manifest is missing columns: {missing}")
    print(f"Manifest rows: {len(rows)}")

    fasta_dir = Path(args.fasta_dir)
    parsed, all_muts, expected_seqs = {}, [], {}

    for r in rows:
        cid = r["candidate_id"].strip()
        wts = split_list(r["wild_type"])
        muts_res = split_list(r["mutation_type"])
        poss = split_list(r["position_idx"])
        try:
            poss = [int(p) for p in poss]
        except ValueError:
            problems[cid].append(f"non-integer position in {r['position_idx']!r}")
            continue

        if not (len(wts) == len(muts_res) == len(poss)):
            problems[cid].append(
                f"list lengths differ: {len(wts)} wt, {len(muts_res)} mut, {len(poss)} pos")
            continue
        try:
            n_declared = int(r["n_mutations"])
        except ValueError:
            n_declared = -1
        if n_declared != len(poss):
            problems[cid].append(f"n_mutations={r['n_mutations']} but {len(poss)} listed")
        if len(set(poss)) != len(poss):
            problems[cid].append("duplicate positions within candidate")
        for w, p, m in zip(wts, poss, muts_res):
            if w == m:
                problems[cid].append(f"no-op mutation {w}{p}{m}")
            if w not in AA or m not in AA:
                problems[cid].append(f"invalid amino-acid letter in {w}{p}{m}")

        muts = list(zip(wts, poss, muts_res))
        all_muts.append(muts)
        exp, errs = apply_mutations(wt, muts, args.offset)
        for e in errs:
            problems[cid].append(e)
        expected_seqs[cid] = exp
        parsed[cid] = (r, muts)

        fpath = fasta_dir / r["fasta_file"].strip()
        if not fpath.exists():
            problems[cid].append(f"FASTA file not found: {fpath}")
            continue
        try:
            _, fseq = read_fasta(fpath)
        except ValueError as e:
            problems[cid].append(f"{fpath.name}: {e}")
            continue
        if set(fseq) - AA:
            problems[cid].append(f"FASTA has invalid characters: {set(fseq) - AA}")
        if len(fseq) != len(wt):
            problems[cid].append(f"FASTA length {len(fseq)} != WT length {len(wt)}")
        elif fseq != exp:
            diffs = [f"idx {i} (pos {i + args.offset}): FASTA={a} expected={b}"
                     for i, (a, b) in enumerate(zip(fseq, exp)) if a != b]
            problems[cid].append("FASTA differs from WT+manifest at: " + "; ".join(diffs))

    # stray FASTA files not in the manifest
    in_manifest = {r["fasta_file"].strip() for r in rows}
    for p in sorted(fasta_dir.glob("*.fasta")):
        if p.name not in in_manifest:
            problems["GLOBAL"].append(f"FASTA file not listed in manifest: {p.name}")

    # duplicate sequences
    by_seq = defaultdict(list)
    for cid, s in expected_seqs.items():
        by_seq[s].append(cid)
    for s, ids in by_seq.items():
        if len(ids) > 1:
            problems["GLOBAL"].append(f"identical sequences: {ids}")

    # controls
    fast_seq, fast_errs = apply_mutations(wt, FAST_PETASE, args.offset)
    for e in fast_errs:
        problems["GLOBAL"].append(f"FAST-PETase control: {e}")
    if fast_seq in by_seq:
        print(f"NOTE: FAST-PETase is already in the shortlist as {by_seq[fast_seq]}")

    # report
    failed = any(problems.values())
    print()
    if failed:
        print("PROBLEMS FOUND:")
        for k, msgs in problems.items():
            for m in msgs:
                print(f"  [{k}] {m}")
        offset_scan(wt, all_muts, args.offset)
    else:
        print(f"All {len(rows)} candidates passed every check.")

    out = Path(args.out_dir)
    if failed and not args.force:
        print("\nNo outputs written (fix the problems, or rerun with --force).")
        sys.exit(1)

    (out / "fasta").mkdir(parents=True, exist_ok=True)
    batch = []
    for r in rows:
        cid = r["candidate_id"].strip()
        if cid in expected_seqs:
            batch.append((cid, expected_seqs[cid]))
    batch.append(("control_WT", wt))
    batch.append(("control_FASTPETase", fast_seq))

    for cid, s in batch:
        (out / "fasta" / f"{cid}.fasta").write_text(f">{cid}\n{s}\n")
    with open(out / "colabfold_batch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence"])
        w.writerows(batch)

    with open(out / "validation_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "n_mutations", "mutations", "status"])
        for cid, (r, muts) in parsed.items():
            label = ";".join(f"{a}{p}{b}" for a, p, b in muts)
            w.writerow([cid, len(muts), label,
                        "PROBLEM" if problems.get(cid) else "OK"])

    print(f"\nWrote {len(batch)} sequences to {out}/ "
          f"(fasta/ folder + colabfold_batch.csv + validation_report.csv)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()