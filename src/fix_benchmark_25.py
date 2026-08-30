"""
fix_benchmark_25.py

One-time migration for data/benchmark_25.csv. The file's mutation_type
column has always held the OLD format -- position number fused into the
letter (e.g. "121D", or "121D;D186H" for combos) -- which the new dataset.py
correctly rejects rather than silently mis-parse. This rebuilds the file
into the clean schema: position_idx as a plain comma-separated list,
wild_type/mutation_type as semicolon-separated PER-POSITION letters.

Wild-type letters are NOT trusted from the old CSV (that column is often
just one letter even for multi-position rows, which can't be right for all
of them) -- they're derived directly from the real, already-verified 6EQE
sequence instead, and cross-checked against whatever the CSV did have.

USAGE (from repo root):
    python fix_benchmark_25.py
Writes data/benchmark_25.csv in place; backs up the original to
data/benchmark_25_OLD_FORMAT_backup.csv first.
"""
import re
import shutil
import pandas as pd

SEQRES = """MET ASN PHE PRO ARG ALA SER ARG LEU MET GLN ALA ALA
VAL LEU GLY GLY LEU MET ALA VAL SER ALA ALA ALA THR
ALA GLN THR ASN PRO TYR ALA ARG GLY PRO ASN PRO THR
ALA ALA SER LEU GLU ALA SER ALA GLY PRO PHE THR VAL
ARG SER PHE THR VAL SER ARG PRO SER GLY TYR GLY ALA
GLY THR VAL TYR TYR PRO THR ASN ALA GLY GLY THR VAL
GLY ALA ILE ALA ILE VAL PRO GLY TYR THR ALA ARG GLN
SER SER ILE LYS TRP TRP GLY PRO ARG LEU ALA SER HIS
GLY PHE VAL VAL ILE THR ILE ASP THR ASN SER THR LEU
ASP GLN PRO SER SER ARG SER SER GLN GLN MET ALA ALA
LEU ARG GLN VAL ALA SER LEU ASN GLY THR SER SER SER
PRO ILE TYR GLY LYS VAL ASP THR ALA ARG MET GLY VAL
MET GLY TRP SER MET GLY GLY GLY GLY SER LEU ILE SER
ALA ALA ASN ASN PRO SER LEU LYS ALA ALA ALA PRO GLN
ALA PRO TRP ASP SER SER THR ASN PHE SER SER VAL THR
VAL PRO THR LEU ILE PHE ALA CYS GLU ASN ASP SER ILE
ALA PRO VAL ASN SER SER ALA LEU PRO ILE TYR ASP SER
MET SER ARG ASN ALA LYS GLN PHE LEU GLU ILE ASN GLY
GLY SER HIS SER CYS ALA ASN SER GLY ASN SER ASN GLN
ALA LEU ILE GLY LYS LYS GLY VAL ALA TRP MET LYS ARG
PHE MET ASP ASN ASP THR ARG TYR SER THR PHE ALA CYS
GLU ASN PRO ASN SER THR ARG VAL SER ASP PHE ARG THR
ALA ASN CYS SER LEU GLU HIS HIS HIS HIS HIS HIS""".split()
T2O = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
       'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
       'THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
REAL_SEQ = [T2O[r] for r in SEQRES]  # index 0 = residue 1

def real_aa(pos):
    return REAL_SEQ[pos - 1] if 1 <= pos <= len(REAL_SEQ) else None

TOKEN_RE = re.compile(r'^([A-Za-z]?)(\d+)([A-Za-z])$')  # optional leading wt letter, position, mutant letter

def parse_old_mutation_field(field):
    """
    Old field is semicolon-separated tokens like "121D" or "D186H" (the
    leading wild-type letter is sometimes present, sometimes not -- both
    forms show up across the file). Returns list of (position, mutant_letter).
    """
    tokens = [t.strip() for t in str(field).split(';') if t.strip()]
    parsed = []
    for tok in tokens:
        m = TOKEN_RE.match(tok)
        if not m:
            raise ValueError(f"Could not parse token '{tok}' in field '{field}'")
        _, pos, mut = m.groups()
        parsed.append((int(pos), mut.upper()))
    return parsed

def migrate():
    path = "data/benchmark_25.csv"
    backup = "data/benchmark_25_OLD_FORMAT_backup.csv"
    shutil.copy(path, backup)
    print(f"Backed up original to {backup}")

    df = pd.read_csv(path)
    new_rows = []
    problems = []

    for idx, row in df.iterrows():
        try:
            parsed = parse_old_mutation_field(row['mutation_type'])
        except ValueError as e:
            problems.append(f"Row {idx}: {e}")
            continue

        positions, mut_letters, wt_letters = [], [], []
        for pos, mut in parsed:
            real = real_aa(pos)
            if real is None:
                problems.append(f"Row {idx}: position {pos} out of range")
                continue
            positions.append(pos)
            mut_letters.append(mut)
            wt_letters.append(real)  # derived from the REAL sequence, not trusted from the old CSV

        if not positions:
            continue

        new_rows.append({
            'wild_type': ';'.join(wt_letters),
            'mutation_type': ';'.join(mut_letters),
            'position_idx': ','.join(str(p) for p in positions),
            'stability_score': row['stability_score'],
        })

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(path, index=False)

    print(f"\nMigrated {len(new_df)}/{len(df)} rows successfully.")
    if problems:
        print(f"\n{len(problems)} problem(s) encountered:")
        for p in problems:
            print(f"  {p}")
    print(f"\nNew format written to {path}. First 5 rows:")
    print(new_df.head().to_string())

if __name__ == "__main__":
    migrate()