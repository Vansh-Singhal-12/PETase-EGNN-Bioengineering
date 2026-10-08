import sys
from Bio.PDB import PDBParser

OFFSET = 28  # ColabFold residue number = real position - 28 (numbering starts at 1)
expected = {160: "SER", 206: "ASP", 237: "HIS", 87: "TYR", 161: "MET", 159: "TRP", 185: "TRP"}

structure = PDBParser(QUIET=True).get_structure("x", sys.argv[1])
chain = next(structure[0].get_chains())

all_ok = True
for real, name in expected.items():
    res = chain[real - OFFSET]
    ok = res.get_resname() == name
    all_ok = all_ok and ok
    print(f"real position {real:>3}: expected {name}, found {res.get_resname()} "
          f"(file residue {real - OFFSET})  {'OK' if ok else 'MISMATCH'}")

og = chain[160 - OFFSET]["OG"].coord
print("Ser160 OG coordinates (x y z):", " ".join(f"{float(v):.2f}" for v in og))
print("ALL OK" if all_ok else "CHECK NUMBERING - tell Claude")