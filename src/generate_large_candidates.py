import argparse
import csv
import itertools
import numpy as np
from Bio.PDB import PDBParser
from src.protein_registry import build_registry

CAUTION_POSITIONS = {90, 168}
BULKY = set("ILVFWMY")


# ---------- Structural geometry ----------

def build_distance_matrix(protein_key="6EQE"):
    registry = build_registry(verbose=False)
    cfg = registry[protein_key]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_key, cfg["pdb_path"])

    coords, res_numbers = [], []
    target_chain = cfg["chain_id"]
    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue
            for residue in chain:
                if residue.has_id("CA") and residue.id[0] == ' ':
                    coords.append(residue["CA"].get_coord())
                    res_numbers.append(residue.id[1])
        break

    coords = np.array(coords)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    pos_to_idx = {pos: i for i, pos in enumerate(res_numbers)}
    return dist_matrix, pos_to_idx


def spatial_penalty(pos_a, mut_a, pos_b, mut_b, dist_matrix, pos_to_idx, proximal_cutoff=8.0):
    if pos_a not in pos_to_idx or pos_b not in pos_to_idx:
        return 0.0
    d = dist_matrix[pos_to_idx[pos_a], pos_to_idx[pos_b]]
    if d > proximal_cutoff:
        return 0.0
    if mut_a in BULKY and mut_b in BULKY:
        return -2.0
    return 0.0


# ---------- Loading blocks ----------

def load_flexibility_map(building_block_pool_csv):
    flex = {}
    with open(building_block_pool_csv) as f:
        for r in csv.DictReader(f):
            pos = int(r["position_idx"].split(",")[0])
            flex[pos] = str(r.get("is_flexible_region")) == "True"
    return flex


def load_singles_as_blocks(building_block_pool_csv):
    blocks = []
    with open(building_block_pool_csv) as f:
        for r in csv.DictReader(f):
            if not r.get("foldx_ddg_raw"):
                continue
            pos = int(r["position_idx"].split(",")[0])
            score = -float(r["foldx_ddg_raw"])
            blocks.append({
                "mutation_map": {pos: (r["wild_type"], r["mutation_type"])},
                "score": score, "size": 1, "source": "single",
            })
    return blocks


def load_multi_blocks(path, score_col, source_label):
    blocks = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                if not r.get(score_col):
                    continue
                pos_list = [int(p) for p in r["position_idx"].split(",")]
                mut_list = r["mutation_type"].split(";")
                wt_list = r["wild_type"].split(";")
                mutation_map = dict(zip(pos_list, zip(wt_list, mut_list)))
                blocks.append({
                    "mutation_map": mutation_map, "score": float(r[score_col]),
                    "size": len(pos_list), "source": source_label,
                })
    except FileNotFoundError:
        print(f"[large_candidates] Note: {path} not found, skipping")
    return blocks


# ---------- Mechanism 1: no-overlap block beam search ----------

def beam_search_blocks(blocks, flex_map, dist_matrix, pos_to_idx, max_size=12,
                        beam_width=10, max_rounds=8, max_rigid_positions=5,
                        caution_penalty=1.5, proximal_cutoff=8.0):
    beam = [(frozenset(), {}, 0.0)]

    for round_num in range(max_rounds):
        candidates = []
        for pos_set, mut_map, score in beam:
            for block in blocks:
                block_positions = set(block["mutation_map"].keys())
                if block_positions & pos_set:
                    continue

                new_size = len(pos_set) + len(block_positions)
                if new_size > max_size:
                    continue

                new_positions = pos_set | block_positions
                n_rigid = sum(1 for p in new_positions if flex_map.get(p, True) is False)
                if n_rigid > max_rigid_positions:
                    continue

                cross_penalty = 0.0
                for p1, (wt1, mut1) in mut_map.items():
                    for p2, (wt2, mut2) in block["mutation_map"].items():
                        cross_penalty += spatial_penalty(p1, mut1, p2, mut2,
                                                           dist_matrix, pos_to_idx, proximal_cutoff)

                caution_adjustment = -caution_penalty * len(block_positions & CAUTION_POSITIONS)
                new_score = score + block["score"] + cross_penalty + caution_adjustment
                new_mut_map = {**mut_map, **block["mutation_map"]}

                candidates.append((new_positions, new_mut_map, new_score))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[2], reverse=True)
        seen, new_beam = set(), []
        for pos_set, mut_map, score in candidates:
            if pos_set in seen:
                continue
            seen.add(pos_set)
            new_beam.append((pos_set, mut_map, score))
            if len(new_beam) >= beam_width:
                break
        beam = new_beam

    return beam


# ---------- Mechanism 2: overlap merge ----------

def try_overlap_merge(a, b, max_size):
    shared = set(a["mutation_map"]) & set(b["mutation_map"])
    if not shared:
        return None  # no overlap -> not this mechanism's job, block search handles that case
    for pos in shared:
        if a["mutation_map"][pos] != b["mutation_map"][pos]:
            return None  # conflict: same position, different mutation
    merged = {**a["mutation_map"], **b["mutation_map"]}
    if len(merged) > max_size or len(merged) <= max(len(a["mutation_map"]), len(b["mutation_map"])):
        return None
    return merged


def generate_overlap_merges(blocks, max_size=12, max_candidates=50):
    # Only consider blocks of size >= 2 for overlap merging -- two singles
    # "overlapping" is meaningless (they'd have to be the same position).
    multi_blocks = [b for b in blocks if b["size"] >= 2]

    merges, seen = [], set()
    for a, b in itertools.combinations(multi_blocks, 2):
        merged_map = try_overlap_merge(a, b, max_size)
        if merged_map is None:
            continue
        key = frozenset(merged_map.items())
        if key in seen:
            continue
        seen.add(key)
        merges.append({
            "mutation_map": merged_map,
            "estimate": (a["score"] or 0) + (b["score"] or 0),  # naive sum, shared positions double-counted
        })

    merges.sort(key=lambda m: m["estimate"], reverse=True)
    return merges[:max_candidates]


# ---------- Runner ----------

def generate_all(building_block_pool_csv, validated_pool_specs, output_path,
                  min_size=6, max_size=12, beam_width=10, max_rounds=8,
                  max_rigid_positions=5, caution_penalty=1.5, proximal_cutoff=8.0,
                  max_merge_candidates=50):
    flex_map = load_flexibility_map(building_block_pool_csv)
    dist_matrix, pos_to_idx = build_distance_matrix()

    blocks = load_singles_as_blocks(building_block_pool_csv)
    for path, score_col, label in validated_pool_specs:
        blocks.extend(load_multi_blocks(path, score_col, label))

    size_counts = {}
    for b in blocks:
        size_counts[b["size"]] = size_counts.get(b["size"], 0) + 1
    print(f"[large_candidates] Loaded {len(blocks)} total blocks: "
          f"{', '.join(f'{k}-pt x{v}' for k, v in sorted(size_counts.items()))}")

    # Mechanism 1
    beam = beam_search_blocks(blocks, flex_map, dist_matrix, pos_to_idx,
                               max_size=max_size, beam_width=beam_width, max_rounds=max_rounds,
                               max_rigid_positions=max_rigid_positions,
                               caution_penalty=caution_penalty, proximal_cutoff=proximal_cutoff)

    # Mechanism 2
    merges = generate_overlap_merges(blocks, max_size=max_size, max_candidates=max_merge_candidates)

    rows = []
    for pos_set, mut_map, score in beam:
        if len(pos_set) < min_size:
            continue
        pos_list = sorted(pos_set)
        rows.append({
            "wild_type": ";".join(mut_map[p][0] for p in pos_list),
            "mutation_type": ";".join(mut_map[p][1] for p in pos_list),
            "position_idx": ",".join(str(p) for p in pos_list),
            "stability_score": 0.0, "protein_id": "6EQE",
            "_source_mechanism": "no_overlap_beam", "_estimate": round(score, 3),
            "_n_mutations": len(pos_list),
        })

    for m in merges:
        pos_list = sorted(m["mutation_map"])
        if len(pos_list) < min_size:
            continue
        rows.append({
            "wild_type": ";".join(m["mutation_map"][p][0] for p in pos_list),
            "mutation_type": ";".join(m["mutation_map"][p][1] for p in pos_list),
            "position_idx": ",".join(str(p) for p in pos_list),
            "stability_score": 0.0, "protein_id": "6EQE",
            "_source_mechanism": "overlap_merge", "_estimate": round(m["estimate"], 3),
            "_n_mutations": len(pos_list),
        })

    seen_keys, deduped = set(), []
    for r in rows:
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(r)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wild_type", "mutation_type", "position_idx",
                                                 "stability_score", "protein_id",
                                                 "_source_mechanism", "_estimate", "_n_mutations"])
        writer.writeheader()
        writer.writerows(deduped)

    n_beam = sum(1 for r in deduped if r["_source_mechanism"] == "no_overlap_beam")
    n_merge = sum(1 for r in deduped if r["_source_mechanism"] == "overlap_merge")
    print(f"\n[large_candidates] {n_beam} from no-overlap beam search, {n_merge} from overlap merge, "
          f"{len(deduped)} total after dedup -> {output_path}")
    for r in sorted(deduped, key=lambda r: r["_estimate"], reverse=True)[:10]:
        print(f"    [{r['_n_mutations']}-pt, {r['_source_mechanism']}, est={r['_estimate']}] "
              f"{r['wild_type']}->{r['mutation_type']} @ {r['position_idx']}")

    print(f"\n[large_candidates] REMINDER: overlap_merge estimates double-count shared positions' "
          f"scores and are LESS realistic than no_overlap_beam estimates -- both are candidate-"
          f"generation signals only. Run FoldX on all {len(deduped)} before drawing any conclusion.")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--building_block_pool", type=str, default="results/building_block_pool.csv")
    ap.add_argument("--output", type=str, default="data/large_combo_candidates.csv")
    ap.add_argument("--min_size", type=int, default=6)
    ap.add_argument("--max_size", type=int, default=12)
    ap.add_argument("--beam_width", type=int, default=10)
    ap.add_argument("--max_rounds", type=int, default=8)
    ap.add_argument("--max_rigid_positions", type=int, default=5)
    ap.add_argument("--caution_penalty", type=float, default=1.5)
    ap.add_argument("--proximal_cutoff", type=float, default=8.0)
    ap.add_argument("--max_merge_candidates", type=int, default=50)
    args = ap.parse_args()

    validated_pool_specs = [
        ("results/validated_pair_pool.csv", "foldx_combo_score", "pair"),
        ("results/validated_triple_pool.csv", "foldx_triple_score", "triple"),
        ("results/validated_quad_pool.csv", "foldx_quad_score", "quad"),
        ("results/validated_quint_pool.csv", "foldx_quint_score", "quint"),
    ]

    generate_all(args.building_block_pool, validated_pool_specs, args.output,
                min_size=args.min_size, max_size=args.max_size, beam_width=args.beam_width,
                max_rounds=args.max_rounds, max_rigid_positions=args.max_rigid_positions,
                caution_penalty=args.caution_penalty, proximal_cutoff=args.proximal_cutoff,
                max_merge_candidates=args.max_merge_candidates)