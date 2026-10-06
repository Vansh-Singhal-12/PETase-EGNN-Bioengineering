import argparse
import csv
import os


def load_wt_sequence(fasta_path):
    with open(fasta_path) as f:
        lines = f.readlines()
    return lines[1].strip()


def build_mutated_sequence(wt_sequence, wt_list, mut_list, pos_list, first_residue=29):
    seq = list(wt_sequence)
    for wt_aa, mut_aa, pos in zip(wt_list, mut_list, pos_list):
        seq_idx = pos - first_residue
        if seq_idx < 0 or seq_idx >= len(seq):
            raise ValueError(f"Position {pos} out of range for sequence of length {len(seq)}")
        if seq[seq_idx] != wt_aa:
            raise ValueError(f"Mismatch at position {pos}: sequence has '{seq[seq_idx]}', "
                              f"candidate claims wild-type '{wt_aa}' -- check data integrity "
                              f"before trusting this candidate's structure prediction.")
        seq[seq_idx] = mut_aa
    return "".join(seq)


def generate_sequences(shortlist_csv, wt_fasta, output_dir, manifest_path, first_residue=29):
    wt_sequence = load_wt_sequence(wt_fasta)

    with open(shortlist_csv) as f:
        rows = list(csv.DictReader(f))

    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    n_errors = 0
    for i, r in enumerate(rows):
        wt_list = r["wild_type"].split(";")
        mut_list = r["mutation_type"].split(";")
        pos_list = [int(p) for p in r["position_idx"].split(",")]

        candidate_id = f"candidate_{i+1:03d}"
        try:
            mutated_seq = build_mutated_sequence(wt_sequence, wt_list, mut_list, pos_list, first_residue)
        except ValueError as e:
            print(f"[gen_sequences] ERROR on {candidate_id} ({r['wild_type']}->{r['mutation_type']} "
                  f"@ {r['position_idx']}): {e}")
            n_errors += 1
            continue

        fasta_filename = f"{candidate_id}.fasta"
        fasta_path = os.path.join(output_dir, fasta_filename)
        with open(fasta_path, "w") as f:
            f.write(f">{candidate_id}\n{mutated_seq}\n")

        manifest.append({
            "candidate_id": candidate_id,
            "fasta_file": fasta_filename,
            "wild_type": r["wild_type"], "mutation_type": r["mutation_type"],
            "position_idx": r["position_idx"],
            "n_mutations": len(pos_list),
            "source_mechanism": r.get("_source_mechanism", ""),
        })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "fasta_file", "wild_type",
                                                 "mutation_type", "position_idx", "n_mutations",
                                                 "source_mechanism"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\n[gen_sequences] Generated {len(manifest)} sequence files in {output_dir}/")
    if n_errors:
        print(f"[gen_sequences] {n_errors} candidates FAILED validation (wild-type mismatch) -- "
              f"fix the underlying data before proceeding, do not silently skip these.")
    print(f"[gen_sequences] Manifest -> {manifest_path}")

    return manifest_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shortlist", type=str, default="data/colabfold_docking_shortlist.csv")
    ap.add_argument("--wt_fasta", type=str, default="data/6EQE_wt_sequence.fasta")
    ap.add_argument("--output_dir", type=str, default="data/colabfold_sequences")
    ap.add_argument("--manifest", type=str, default="data/colabfold_manifest.csv")
    ap.add_argument("--first_residue", type=int, default=29)
    args = ap.parse_args()

    generate_sequences(args.shortlist, args.wt_fasta, args.output_dir, args.manifest, args.first_residue)