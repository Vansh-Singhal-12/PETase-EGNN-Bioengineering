import re
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from Bio.PDB import PDBParser

from src.protein_registry import build_registry

# Explicit mapping instead of Bio.PDB.Polypeptide.three_to_one -- that
# function was removed in newer Biopython versions (caught by this
# project's own smoke test), so this avoids depending on an internal API
# that can silently disappear across environments/versions.
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
    'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}


def three_to_one(resname):
    if resname not in THREE_TO_ONE:
        raise KeyError(resname)
    return THREE_TO_ONE[resname]


AA_PROPERTIES = {
    'A': [89.1, 1.8, 0, 0], 'R': [174.2, -4.5, 1, 4], 'N': [132.1, -3.5, 0, 2],
    'D': [133.1, -3.5, -1, 2], 'C': [121.2, 2.5, 0, 0], 'E': [147.1, -3.5, -1, 2],
    'Q': [146.1, -3.5, 0, 2], 'G': [75.1, -0.4, 0, 0], 'H': [155.2, -3.2, 0.5, 2],
    'I': [131.2, 4.5, 0, 0], 'L': [131.2, 3.8, 0, 0], 'K': [146.2, -3.9, 1, 2],
    'M': [149.2, 1.9, 0, 0], 'F': [165.2, 2.8, 0, 0], 'P': [115.1, -1.6, 0, 0],
    'S': [105.1, -0.8, 0, 1], 'T': [119.1, -0.7, 0, 1], 'W': [204.2, -0.9, 0, 1],
    'Y': [181.2, -1.3, 0, 1], 'V': [117.1, 4.2, 0, 0]
}
props_matrix = np.array(list(AA_PROPERTIES.values()))
props_mean = props_matrix.mean(axis=0)
props_std = props_matrix.std(axis=0) + 1e-8
AA_PROPERTIES_NORM = {
    aa: ((np.array(props) - props_mean) / props_std).tolist()
    for aa, props in AA_PROPERTIES.items()
}

# Handles both plain positions ("121") and PDB insertion-code positions
# ("27B", "27C") -- the 4 S2648 rows that got silently dropped earlier
# (1LVE L27C, V27B, Y27D) are legitimate data, not malformed rows; this
# fixes that instead of permanently excluding them.
POS_RE = re.compile(r'^(\d+)([A-Za-z]?)$')


class PETaseMutationDataset:
    """
    Multi-protein dataset. Every row's CSV must resolve to a `protein_key`
    matching a registry entry ("6EQE", "LCC", or "{PDBID}_{CHAIN}" for
    S2648 structures). Rows for different proteins use fully independent
    graphs -- no cross-protein leakage is structurally possible.

    `source_tag` distinguishes real experimental rows from synthetic
    combination rows generated during augmentation, so pretraining can use
    both while fine-tuning/eval can filter to real-only.
    """

    def __init__(self, csv_paths, augment_inverse=True, augment_combinations=False,
                 max_synthetic_ratio=1.0, registry=None):
        """
        csv_paths: single path or list of paths. Each CSV needs columns:
            wild_type, mutation_type, position_idx, stability_score,
            and EITHER protein_id (for "6EQE"/"LCC") OR protein_id+chain
            (for S2648 rows, will be combined into "{protein_id}_{chain}").
        augment_combinations: only meaningful for PRETRAINING -- generates
            synthetic multi-point combos via additive-approximation labels.
            Must stay False for any fine-tuning or evaluation dataset,
            since the additive assumption is known to be wrong in real
            cases (verified session 5: Stevensen et al. combo ΔΔG did not
            equal the sum of its parts).
        max_synthetic_ratio: CAPS synthetic combo rows per protein to at
            most this multiple of that protein's REAL row count (default
            1.0 = synthetic rows never outnumber real rows for any given
            protein). This exists because of a real, diagnosed failure:
            an earlier run with this cap absent generated 8428 synthetic
            combo rows against only 2629 real rows (61% of the whole
            pretraining set), and the model learned to fit the trivially-
            exploitable "sum of parts" pattern in the synthetic majority
            rather than genuine per-mutation structural signal -- confirmed
            by a fresh in-sample eval showing rho=0.26 on real rows despite
            training logs reporting rho=0.85 on the full (synthetic-
            dominated) mix. Capping the ratio keeps combo-graph exposure
            for the model without letting it dominate the loss.
        """
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        self.registry = registry if registry is not None else build_registry(verbose=False)
        self.max_synthetic_ratio = max_synthetic_ratio

        frames = []
        for path in csv_paths:
            df = pd.read_csv(path)
            if "protein_id" not in df.columns:
                # Pre-multi-protein CSVs (mutations_verified_stability.csv,
                # benchmark_25.csv, mutations_clean.csv) predate this column
                # entirely -- every one of them is 6EQE-specific data from
                # earlier sessions, so that's the correct, documented default
                # rather than a silent guess.
                df["protein_id"] = "6EQE"
            if "chain" in df.columns:
                df["protein_key"] = df.apply(
                    lambda r: f"{str(r['protein_id']).upper()}_{r['chain']}"
                    if pd.notna(r.get("chain")) else str(r["protein_id"]).upper(),
                    axis=1)
            else:
                df["protein_key"] = df["protein_id"].astype(str).str.upper()
            frames.append(df)
        self.df = pd.concat(frames, ignore_index=True)

        unknown = set(self.df["protein_key"]) - set(self.registry.keys())
        if unknown:
            print(f"[dataset] WARNING: {len(unknown)} protein_key(s) in the CSV have no "
                  f"registry entry (missing PDB file?) -- dropping their rows: {sorted(unknown)[:10]}...")
            self.df = self.df[self.df["protein_key"].isin(self.registry.keys())].reset_index(drop=True)

        self.augment_inverse = augment_inverse
        self.augment_combinations = augment_combinations

        self.base_graphs = {}
        for protein_key in self.df["protein_key"].unique():
            self.base_graphs[protein_key] = self._load_protein_as_graph(protein_key)

        self.items = self._build_dataset_items()

    def _load_protein_as_graph(self, protein_key):
        cfg = self.registry[protein_key]
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(protein_key, cfg["pdb_path"])

        coords, res_names = [], []
        target_chain = cfg["chain_id"]
        for model in structure:
            for chain in model:
                if chain.id != target_chain:
                    continue
                for residue in chain:
                    if residue.has_id("CA") and residue.id[0] == ' ':
                        coords.append(residue["CA"].get_coord())
                        res_names.append(residue.get_resname())
            break

        coords = np.array(coords)
        num_nodes = len(coords)

        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        edge_indices = np.where((dist_matrix <= 8.0) & (dist_matrix > 0))
        edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)

        x_feat = []
        for resname in res_names:
            try:
                one_letter = three_to_one(resname)
            except Exception:
                one_letter = 'A'
            wt_props = AA_PROPERTIES_NORM.get(one_letter, [0.0, 0.0, 0.0, 0.0])
            x_feat.append(wt_props + [0.0, 0.0, 0.0, 0.0])

        x_tensor = torch.tensor(x_feat, dtype=torch.float)
        pos_tensor = torch.tensor(coords, dtype=torch.float)

        triad = cfg.get("catalytic_triad")
        if triad:
            first_res = cfg["first_resolved_residue"]
            cat_idx = [r - first_res for r in triad]
            cat_idx = [i for i in cat_idx if 0 <= i < num_nodes]
            cat_coords = coords[cat_idx]
            dists_to_cat = np.linalg.norm(coords[:, None, :] - cat_coords[None, :, :], axis=-1)
            min_cat_dists = dists_to_cat.min(axis=-1)
            active_site_shield = torch.tensor(min_cat_dists <= 10.0, dtype=torch.bool)
        else:
            # No verified catalytic triad for this protein -- shield mask is
            # all-False, so the shield loss term contributes exactly zero
            # for these rows (train.py already guards on mask.sum() > 0).
            active_site_shield = torch.zeros(num_nodes, dtype=torch.bool)

        graph = Data(x=x_tensor, pos=pos_tensor, edge_index=edge_index)
        graph.active_site_shield = active_site_shield
        return graph

    def _parse_letter_list(self, field, n_positions):
        """
        Parses a wild_type or mutation_type field into a list aligned 1:1
        with a row's positions. Supports semicolon-separated per-position
        letters ("I;S" for a 2-position row -- the format the real
        I168R/S188Q entry already uses), or a single letter broadcast to
        all positions (only valid/correct when n_positions == 1 -- this is
        what every single-mutation row in every CSV so far actually is).
        """
        field = str(field)
        if ';' in field:
            parts = [p.strip() for p in field.split(';') if p.strip()]
        else:
            parts = [field.strip()] * n_positions
        if len(parts) != n_positions:
            raise ValueError(
                f"wild_type/mutation_type field '{field}' has {len(parts)} letter(s) "
                f"but the row has {n_positions} position(s) -- these must match "
                f"exactly (use ';'-separated letters per position for multi-point rows)."
            )
        for p in parts:
            if p not in AA_PROPERTIES_NORM:
                raise ValueError(
                    f"'{p}' (parsed from field '{field}') is not a valid single-letter "
                    f"amino acid code. This usually means the CSV is using the OLD "
                    f"format where mutation_type held strings like '121D;D186H' "
                    f"(position number fused with the letter) instead of clean "
                    f"per-position letters like 'D;H'. Fix the CSV, not this parser."
                )
        return parts

    def _map_pos_to_node_idx(self, pos_str, protein_key):
        cfg = self.registry[protein_key]
        first_res = cfg["first_resolved_residue"]
        m = POS_RE.match(str(pos_str).strip())
        if not m:
            return None
        base_pos = int(m.group(1))
        # insertion code (e.g. the "B" in "27B") -- treated as the same base
        # residue's node for graph purposes, since Cα-graph resolution can't
        # distinguish sub-residue insertion positions anyway.
        node_idx = base_pos - first_res if base_pos >= first_res else base_pos
        num_nodes = self.base_graphs[protein_key].x.size(0)
        if node_idx < 0 or node_idx >= num_nodes:
            return None
        return node_idx

    def _parse_pos_string(self, pos_str, protein_key):
        raw_list = [p.strip() for p in str(pos_str).replace('[', '').replace(']', '').split(',') if p.strip()]
        mapped = [self._map_pos_to_node_idx(p, protein_key) for p in raw_list]
        mapped = [m for m in mapped if m is not None]
        return mapped if mapped else None

    def _build_dataset_items(self):
        items = []
        raw_rows = []

        for idx, row in self.df.iterrows():
            protein_key = row["protein_key"]
            pos_list = self._parse_pos_string(row['position_idx'], protein_key)
            if pos_list is None:
                continue
            score = float(row['stability_score'])
            try:
                mut_list = self._parse_letter_list(row.get('mutation_type', 'A'), len(pos_list))
                wt_list = self._parse_letter_list(row.get('wild_type', 'A'), len(pos_list))
            except ValueError as e:
                print(f"[dataset] WARNING: skipping row {idx} ({protein_key}): {e}")
                continue

            r = {'pos_list': pos_list, 'score': score, 'mut_list': mut_list,
                 'wt_list': wt_list, 'protein_key': protein_key}
            raw_rows.append(r)
            items.append((pos_list, score, wt_list, mut_list, protein_key, "real"))

        if self.augment_inverse:
            for r in raw_rows:
                items.append((r['pos_list'], -r['score'], r['wt_list'], r['mut_list'],
                              r['protein_key'], "real_inverse"))

        if self.augment_combinations:
            # Builds synthetic 2, 3, 4, and 5-point combinations, matching
            # the combo sizes actually present in benchmark_25.csv (up to
            # 5 mutations), so pretraining exposes the model to graphs with
            # as many simultaneously-mutated nodes as it'll see at eval time.
            # Score is a PLAIN additive sum -- no synergy multiplier, unlike
            # the original PETase-specific code, since any multiplier tuned
            # to a handful of famous PETase variants has no justified basis
            # for arbitrary S2648 proteins. This is explicitly a rough
            # approximation for pretraining only (see class docstring).
            #
            # CAPPED per protein at max_synthetic_ratio * (that protein's
            # real row count) -- see __init__ docstring for why this cap
            # exists. Without it, synthetic rows can vastly outnumber real
            # ones (diagnosed: 8428 synthetic vs 2629 real, 61% of the
            # dataset), letting the model fit the trivial "sum of parts"
            # shortcut instead of learning real per-mutation signal.
            by_protein = {}
            for r in raw_rows:
                by_protein.setdefault(r['protein_key'], []).append(r)

            offsets_by_depth = {2: [17], 3: [11, 23], 4: [7, 19, 31], 5: [5, 13, 29, 37]}
            total_synthetic_added = 0

            for protein_key, rows in by_protein.items():
                n = len(rows)
                budget = max(0, int(round(n * self.max_synthetic_ratio)))
                added_for_this_protein = 0
                if budget == 0:
                    continue

                for depth, offsets in offsets_by_depth.items():
                    if added_for_this_protein >= budget:
                        break
                    if n <= depth:
                        continue
                    for i in range(n):
                        if added_for_this_protein >= budget:
                            break
                        combo_rows = [rows[i]] + [rows[(i + off) % n] for off in offsets]
                        # skip if any two picks share a position -- can't
                        # cleanly represent two different mutations at the
                        # same node in one graph
                        all_pos = [p for cr in combo_rows for p in cr['pos_list']]
                        if len(set(all_pos)) != len(all_pos):
                            continue
                        comb_pos, comb_wt, comb_mt = [], [], []
                        for cr in combo_rows:
                            comb_pos.extend(cr['pos_list'])
                            comb_wt.extend(cr['wt_list'])
                            comb_mt.extend(cr['mut_list'])
                        comb_score = sum(cr['score'] for cr in combo_rows)
                        items.append((comb_pos, comb_score, comb_wt, comb_mt,
                                      protein_key, f"synthetic_combo_{depth}pt"))
                        added_for_this_protein += 1

                total_synthetic_added += added_for_this_protein

            n_real = len(raw_rows)
            n_inverse = len(raw_rows) if self.augment_inverse else 0
            print(f"[dataset] Row composition: {n_real} real, {n_inverse} inverse-augmented, "
                  f"{total_synthetic_added} synthetic combo (capped at {self.max_synthetic_ratio}x "
                  f"real count per protein) -- synthetic is "
                  f"{100 * total_synthetic_added / max(1, n_real + n_inverse + total_synthetic_added):.1f}% "
                  f"of the total.")

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pos_list, score, wt_list, mut_list, protein_key, source_tag = self.items[idx]

        graph = self.base_graphs[protein_key].clone()
        num_nodes = graph.x.size(0)

        # Per-position delta -- each mutated node gets ITS OWN wild-type ->
        # mutant delta, not one delta copied across every position (that
        # was the bug this replaces).
        for p, w_code, m_code in zip(pos_list, wt_list, mut_list):
            if p >= num_nodes:
                continue
            m_props = np.array(AA_PROPERTIES_NORM.get(m_code, [0.0, 0.0, 0.0, 0.0]))
            w_props = np.array(AA_PROPERTIES_NORM.get(w_code, [0.0, 0.0, 0.0, 0.0]))
            delta_props = (m_props - w_props).tolist()
            graph.x[p, 4:] = torch.tensor(delta_props, dtype=torch.float)

        target_tensor = torch.tensor([score], dtype=torch.float)
        mutation_pos_tensor = torch.tensor(pos_list, dtype=torch.long)
        shield_mask = graph.active_site_shield

        return graph, target_tensor, mutation_pos_tensor, shield_mask, protein_key, source_tag