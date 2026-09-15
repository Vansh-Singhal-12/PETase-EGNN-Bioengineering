"""
Building-block pool: a UNION of four independently-justified groups, not an
averaged/blended score. This replaces an earlier percentile-averaged
"consensus_score" approach, which was found to systematically bury strong
FoldX-validated stability wins (e.g., D186V, +12.91 degC real Tm increase)
under ESM's evolutionary-plausibility penalty -- because industrially-
motivated stability jumps beyond what wild IsPETase ever needed in its
natural environment are EXPECTED to score low on "evolutionary favorability"
even when they are exactly the mutations we want. ESM should only ever ADD
candidates FoldX may have missed (a rescue role), never suppress a strong
FoldX winner.

Four groups, unioned together:
  1. foldx_primary      -- top N candidates purely by FoldX ddG, no ESM input.
  2. esm_rescue          -- top N candidates purely by ESM score, excluding
                             anything already in foldx_primary. Catches cases
                             FoldX may have gotten wrong (the S121E-type case).
  3. literature_hotspot  -- the EXACT real, literature-verified stabilizing
                             mutations (not every substitution at that
                             position), force-included regardless of score.
  4. research_hotspot    -- positions literature flags as MECHANISTICALLY
                             important (e.g. Loop 10 region, the D186 5A
                             contact shell, newly-identified thermostability
                             positions from semi-rational design papers)
                             WITHOUT a single verified best substitution.
                             Only the single BEST-SCORING FoldX candidate at
                             each such position is included, not all 19
                             substitutions, to add structural coverage
                             without reintroducing untested noise. A lenient
                             plausibility floor (FoldX-flipped > -2.0) keeps
                             out anything clearly destabilizing.

is_flexible_region (from the B-factor pass) is carried through as an
informational column on every row -- context/tiebreaker later, not a
fifth numerically-blended score.
"""
import argparse
import csv


KNOWN_REAL_MUTATIONS = {
    ("L", "F", 117), ("Q", "Y", 119), ("S", "E", 136),
    ("I", "R", 168), ("S", "Q", 188), ("D", "V", 186), ("D", "N", 186),
    ("S", "E", 121), ("R", "A", 280), ("R", "Q", 224), ("N", "K", 233),
    ("S", "F", 238), ("G", "A", 165), ("S", "H", 214),
}

RESEARCH_HOTSPOT_POSITIONS = {
    186, 187, 188, 189, 190, 191,          # Loop 10 region (D186 mechanism paper)
    120, 161, 164, 168, 184, 185, 214, 218,  # 5A shell around D186
    114, 205, 233, 269,                     # semi-rational design paper, newly-identified positions
}


def build_building_block_pool(input_csv_path, output_path,
                                n_foldx_primary=120, n_esm_rescue=30,
                                research_hotspot_floor=-2.0):
    with open(input_csv_path) as f:
        rows = list(csv.DictReader(f))

    parsed_rows = []
    n_skipped = 0
    for r in rows:
        try:
            r["_foldx_flipped"] = -float(r["foldx_ddg_raw"])
        except (ValueError, KeyError):
            n_skipped += 1
            continue
        r["_esm"] = None
        if r.get("esm_score") not in ("", "MISMATCH", None):
            try:
                r["_esm"] = float(r["esm_score"])
            except ValueError:
                pass
        r["_pos"] = int(r["position_idx"].split(",")[0])
        parsed_rows.append(r)

    if n_skipped:
        print(f"[building_block_pool] Skipped {n_skipped} rows with missing/invalid FoldX score")

    # 1. FoldX PRIMARY
    foldx_sorted = sorted(parsed_rows, key=lambda r: r["_foldx_flipped"], reverse=True)
    foldx_primary = foldx_sorted[:n_foldx_primary]
    foldx_primary_keys = {(r["wild_type"], r["mutation_type"], r["position_idx"]) for r in foldx_primary}
    for r in foldx_primary:
        r["_category"] = "foldx_primary"

    # 2. ESM RESCUE
    esm_candidates = [r for r in parsed_rows if r["_esm"] is not None]
    esm_sorted = sorted(esm_candidates, key=lambda r: r["_esm"], reverse=True)
    esm_rescue = []
    for r in esm_sorted:
        if len(esm_rescue) >= n_esm_rescue:
            break
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        if key in foldx_primary_keys:
            continue
        r["_category"] = "esm_rescue"
        esm_rescue.append(r)

    # 3. LITERATURE HOTSPOT (exact verified mutations only)
    existing_keys = foldx_primary_keys | {(r["wild_type"], r["mutation_type"], r["position_idx"]) for r in esm_rescue}
    hotspot_added = []
    for r in parsed_rows:
        match_key = (r["wild_type"], r["mutation_type"], r["_pos"])
        if match_key not in KNOWN_REAL_MUTATIONS:
            continue
        key = (r["wild_type"], r["mutation_type"], r["position_idx"])
        if key in existing_keys:
            continue
        r["_category"] = "literature_hotspot"
        hotspot_added.append(r)
        existing_keys.add(key)

    # 4. RESEARCH HOTSPOT (best-scoring substitution per mechanistically-flagged position)
    existing_keys |= {(r["wild_type"], r["mutation_type"], r["position_idx"]) for r in hotspot_added}
    research_hotspot_added = []
    for pos in RESEARCH_HOTSPOT_POSITIONS:
        candidates_at_pos = [r for r in parsed_rows if r["_pos"] == pos]
        if not candidates_at_pos:
            continue
        best_at_pos = max(candidates_at_pos, key=lambda r: r["_foldx_flipped"])
        key = (best_at_pos["wild_type"], best_at_pos["mutation_type"], best_at_pos["position_idx"])
        if key in existing_keys:
            continue
        if best_at_pos["_foldx_flipped"] < research_hotspot_floor:
            continue
        best_at_pos["_category"] = "research_hotspot"
        research_hotspot_added.append(best_at_pos)
        existing_keys.add(key)

    pool = foldx_primary + esm_rescue + hotspot_added + research_hotspot_added

    fieldnames = ["wild_type", "mutation_type", "position_idx", "foldx_ddg_raw",
                  "esm_score", "is_flexible_region", "category"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in pool:
            writer.writerow({
                "wild_type": r["wild_type"],
                "mutation_type": r["mutation_type"],
                "position_idx": r["position_idx"],
                "foldx_ddg_raw": r.get("foldx_ddg_raw", ""),
                "esm_score": r.get("esm_score", ""),
                "is_flexible_region": r.get("is_flexible_region", ""),
                "category": r["_category"],
            })

    print(f"[building_block_pool] FoldX primary: {len(foldx_primary)}")
    print(f"[building_block_pool] ESM rescue (not already in FoldX primary): {len(esm_rescue)}")
    print(f"[building_block_pool] Literature hotspots (not already included): {len(hotspot_added)}")
    print(f"[building_block_pool] Research hotspots (best per position, not already included): {len(research_hotspot_added)}")
    print(f"[building_block_pool] TOTAL pool: {len(pool)} -> {output_path}")

    print(f"\n[building_block_pool] TOP 15 by FoldX (primary group):")
    for r in foldx_primary[:15]:
        flex = " [FLEXIBLE]" if str(r.get("is_flexible_region")) == "True" else ""
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | FoldX ddG: {r['foldx_ddg_raw']}{flex}")

    print(f"\n[building_block_pool] LITERATURE HOTSPOTS INCLUDED:")
    for r in hotspot_added:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | "
              f"FoldX ddG: {r['foldx_ddg_raw']} | ESM: {r['esm_score']}")

    print(f"\n[building_block_pool] RESEARCH HOTSPOTS INCLUDED (best FoldX candidate per position):")
    for r in research_hotspot_added:
        print(f"    {r['wild_type']}->{r['mutation_type']} @ {r['position_idx']} | FoldX ddG: {r['foldx_ddg_raw']}")

    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/foldx_bfactor_esm_ranked.csv")
    ap.add_argument("--output", type=str, default="results/building_block_pool.csv")
    ap.add_argument("--n_foldx_primary", type=int, default=120)
    ap.add_argument("--n_esm_rescue", type=int, default=30)
    ap.add_argument("--research_hotspot_floor", type=float, default=-2.0)
    args = ap.parse_args()

    build_building_block_pool(args.input, args.output,
                                n_foldx_primary=args.n_foldx_primary,
                                n_esm_rescue=args.n_esm_rescue,
                                research_hotspot_floor=args.research_hotspot_floor)