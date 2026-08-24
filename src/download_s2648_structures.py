"""
download_s2648_structures.py

Downloads all 131 unique PDB structures used in the S2648 dataset. 
Saves everything to data/s2648_structures/<PDBID>.pdb
"""
import os
import time
import urllib.request

PDB_IDS = "1A43,1A5E,1AEP,1AG2,1AJ3,1AKY,1AM7,1AMQ,1ANK,1AON,1APS,1ARR,1AZP,1B26,1B8E,1BLC,1BNI,1BOY,1BTA,1BVC,1C9O,1CAH,1CEY,1CHK,1CSE,1CSP,1CUN,1DKT,1E65,1EY0,1FKJ,1FNA,1FTG,1FVK,1G4I,1G6N,1H7M,1HFZ,1HK0,1HME,1HMK,1HMS,1HTI,1HUU,1IET,1IFC,1IGV,1IHB,1IMQ,1IO2,1IRO,1JIW,1K9Q,1KCQ,1KDX,1KE4,1KFW,1LBI,1LNI,1LUC,1LVE,1LZ1,1MBG,1MGR,1MJC,1MSI,1N0J,1OH0,1OIA,1ONC,1OTR,1P2P,1PDO,1PGA,1POH,1QGV,1QLP,1QM4,1QND,1RG8,1RHG,1RIS,1RN1,1ROP,1RTB,1RTP,1SAK,1SHF,1SHG,1SSO,1SUP,1TEN,1TIT,1TPK,1TTQ,1TYV,1UBQ,1UZC,1VQB,1WIT,1WQ5,1YU5,1YYJ,1ZG4,1ZNJ,2A01,2A36,2ABD,2CI2,2DRI,2H61,2HPR,2IMM,2LZM,2NVH,2OCJ,2RN2,2TRT,2TRX,2TS1,3ECA,3GLY,3HHR,3MBP,3PGK,3SIL,3SSI,4LYZ,5CRO,5DFR,5PTI".split(",")

OUT_DIR = "data/s2648_structures"

def download_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    failed = []
    for i, pdb_id in enumerate(PDB_IDS, 1):
        out_path = os.path.join(OUT_DIR, f"{pdb_id}.pdb")
        if os.path.exists(out_path):
            print(f"[{i}/{len(PDB_IDS)}] {pdb_id} already downloaded, skipping.")
            continue
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"[{i}/{len(PDB_IDS)}] {pdb_id} OK")
        except Exception as e:
            print(f"[{i}/{len(PDB_IDS)}] {pdb_id} FAILED: {e}")
            failed.append(pdb_id)
        time.sleep(0.3)  # be polite to RCSB's servers

    print(f"\nDone. {len(PDB_IDS) - len(failed)}/{len(PDB_IDS)} downloaded successfully.")
    if failed:
        print(f"Failed: {failed}")
        print("(A few of these might be legacy/obsolete IDs needing a redirect --")
        print(" check https://www.rcsb.org/structure/<ID> manually for any that failed.)")

if __name__ == "__main__":
    download_all()