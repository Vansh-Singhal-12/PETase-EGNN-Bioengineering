"""
Splits a combo CSV into N-sized chunks, each written as its own file, so
FoldX can be run on manageable batches across multiple sessions instead of
one long uninterruptible run. Each chunk keeps its own row order intact,
so later parsing/epistasis scripts pair cleanly with that chunk's own
Average_*.fxout -- no cross-chunk offset tracking needed.
"""
import argparse
import csv
import math


def split_csv(input_path, chunk_size, output_prefix):
    with open(input_path) as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    n_chunks = math.ceil(len(rows) / chunk_size)
    print(f"[split] {len(rows)} total rows -> {n_chunks} chunks of up to {chunk_size} each")

    chunk_paths = []
    for i in range(n_chunks):
        chunk_rows = rows[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = f"{output_prefix}_chunk{i+1}.csv"
        with open(chunk_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunk_rows)
        chunk_paths.append(chunk_path)
        print(f"    Chunk {i+1}: {len(chunk_rows)} rows -> {chunk_path}")

    return chunk_paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/screening_round2_pairs_v2.csv")
    ap.add_argument("--chunk_size", type=int, default=4000)
    ap.add_argument("--output_prefix", type=str, default="data/round2_pairs")
    args = ap.parse_args()

    split_csv(args.input, args.chunk_size, args.output_prefix)