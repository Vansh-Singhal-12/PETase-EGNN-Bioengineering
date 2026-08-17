import os
import pandas as pd

# List of position indices used across famous benchmark variants (FAST-PETase, DuraPETase, HotPETase, ThermoPETase)
# These represent strict benchmark positions that MUST NOT appear in the training set.
BENCHMARK_PROTECTED_POSITIONS = {
    121, 186, 224, 233, 280,  # FAST-PETase variants
    159, 132, 214, 238,        # DuraPETase / HotPETase additions
    95,  181,                  # Lu et al. calibration benchmarks
    140, 160, 165              # ADDED: these are used in benchmark_25.csv (T140C,
                                # S160C, G165A) but were missing from this set, so
                                # training rows at these positions were slipping
                                # through the homology filter uncaught.
}

def purge_homologous_training_rows(csv_path="data/mutations.csv", output_path="data/mutations_clean.csv"):
    """
    Purges any mutation row in the training dataset that targets protected benchmark positions,
    preventing data leakage into the ground-truth benchmark.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV path {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # Filter out rows where position_idx is in the benchmark protected list
    clean_df = df[~df['position_idx'].isin(BENCHMARK_PROTECTED_POSITIONS)].copy()
    purged_count = initial_count - len(clean_df)
    
    clean_df.to_csv(output_path, index=False)
    
    print(f"[Homology Filter] Cleaned dataset written to: {output_path}")
    print(f"[Homology Filter] Initial Rows: {initial_count} | Purged Rows: {purged_count} | Remaining Clean Rows: {len(clean_df)}")
    return output_path

if __name__ == "__main__":
    purge_homologous_training_rows()