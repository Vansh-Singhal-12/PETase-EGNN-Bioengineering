import argparse
import csv


def is_flagged(row):
    return str(row.get("drift_flag", "")).strip() == "True"


def find_clean_replacement(original, pool, already_selected_keys, score_tolerance):
    orig_score = original["_score"]
    candidates_in_range = []
    for r in pool:
        if r["_score"] is None:
            continue
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        if key in already_selected_keys:
            continue
        if abs(r["_score"] - orig_score) <= score_tolerance and not is_flagged(r):
            candidates_in_range.append(r)

    if not candidates_in_range:
        return None
    # Prefer the closest-scoring clean candidate, not just any in range
    candidates_in_range.sort(key=lambda r: abs(r["_score"] - orig_score))
    return candidates_in_range[0]


def select_and_merge(quint_pool_csv, large_combos_csv, output_path, n_top_quints=6,
                      score_tolerance=0.5):
    with open(quint_pool_csv) as f:
        reader = csv.DictReader(f)
        quint_fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        quint_rows = list(reader)

    for r in quint_rows:
        r["_score"] = float(r["foldx_quint_score"]) if r.get("foldx_quint_score") else None
        r["_epistasis"] = float(r["foldx_quint_epistasis"]) if r.get("foldx_quint_epistasis") else None

    scored = [r for r in quint_rows if r["_score"] is not None]
    scored.sort(key=lambda r: r["_score"], reverse=True)

    selected = []
    selected_keys = set()
    n_swapped = 0

    for r in scored:
        if len(selected) >= n_top_quints:
            break
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        if key in selected_keys:
            continue

        if is_flagged(r):
            replacement = find_clean_replacement(r, scored, selected_keys, score_tolerance)
            if replacement is not None:
                rep_key = (replacement["wild_type"], replacement["mutation_type"], replacement["position_idx"])
                print(f"[select_quints] SWAP: {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} "
                      f"(score={r['_score']:.3f}, FLAGGED) -> "
                      f"{replacement['wild_type']}->{replacement['mutation_type']} @ {replacement['position_idx']} "
                      f"(score={replacement['_score']:.3f}, clean)")
                selected.append(replacement)
                selected_keys.add(rep_key)
                n_swapped += 1
                continue
            else:
                print(f"[select_quints] KEEP (flagged, no clean alternative within ±{score_tolerance}): "
                      f"{r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} (score={r['_score']:.3f})")

        selected.append(r)
        selected_keys.add(key)

    print(f"\n[select_quints] Final top {len(selected)} quints ({n_swapped} swapped for cleaner alternatives):")
    for r in selected:
        epi_note = ""
        if r["_epistasis"] is not None and r["_epistasis"] > 1.0:
            epi_note = f"   [strong epistasis: {r['_epistasis']:.3f}]"
        flag_note = " [DRIFT-FLAGGED]" if is_flagged(r) else ""
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"score: {r['_score']:.3f}{flag_note}{epi_note}")

    with open(large_combos_csv) as f:
        reader = csv.DictReader(f)
        large_fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        large_rows = list(reader)

    # Clean temporary internal tracking keys before writing output
    cleaned_selected = []
    for r in selected:
        clean_row = {k: v for k, v in r.items() if not k.startswith("_")}
        cleaned_selected.append(clean_row)

    # Combine fieldnames in order of appearance across both CSV files
    all_fieldnames = list(dict.fromkeys(large_fieldnames + quint_fieldnames))

    combined = large_rows + cleaned_selected

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, restval="")
        writer.writeheader()
        writer.writerows(combined)

    print(f"\n[select_quints] Merged {len(large_rows)} large combos + {len(cleaned_selected)} quints "
          f"= {len(combined)} total -> {output_path}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quint_pool", type=str, default="results/validated_quint_pool.csv")
    ap.add_argument("--large_combos", type=str, default="results/large_combos_master_table.csv")
    ap.add_argument("--output", type=str, default="data/colabfold_docking_shortlist.csv")
    ap.add_argument("--n_top_quints", type=int, default=6)
    ap.add_argument("--score_tolerance", type=float, default=0.5)
    args = ap.parse_args()

    select_and_merge(args.quint_pool, args.large_combos, args.output,
                     n_top_quints=args.n_top_quints, score_tolerance=args.score_tolerance)