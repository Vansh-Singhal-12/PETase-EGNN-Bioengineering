import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN

def evaluate_phase2_benchmark(pdb_path="data/6eqe.pdb", benchmark_csv="data/benchmark_25.csv", checkpoint_path="checkpoints/best_egnn.pt"):
    """
    Evaluates the trained E(3)-EGNN checkpoint against ground-truth literature variants.
    """
    print("=" * 65)
    print("GROUND-TRUTH HISTORICAL BENCHMARK EVALUATION")
    print("=" * 65)
    
    if not os.path.exists(benchmark_csv):
        print(f"Benchmark CSV '{benchmark_csv}' not found. Please populate with the 25 literature variants.")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint '{checkpoint_path}' not found. Train the model first.")
        return

    # Load ground-truth benchmark CSV
    df_benchmark = pd.read_csv(benchmark_csv)
    expected_length = len(df_benchmark)
    
    # --- Data-Length Test Enforcement ---
    print(f"Loaded {expected_length} benchmark rows from '{benchmark_csv}'")
    
    # Instantiate dataset and model
    dataset = PETaseMutationDataset(pdb_path=pdb_path, csv_path=benchmark_csv)
    
    # Data-Length Test Assertion
    assert len(dataset) == expected_length, f"[DATA-LENGTH TEST FAILED] Expected {expected_length} rows, but dataset yielded {len(dataset)}."
    print("[DATA-LENGTH TEST PASSED] Verification: Prediction array length matches benchmark dataset.")

    # Load Model Checkpoint
    model = PETaseStabilityEGNN(num_amino_acids=4, emb_dim=32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph_data, target_score, mutation_pos, _ = dataset[idx]
            all_preds = model(graph_data)
            
            pred_val = all_preds[mutation_pos].item()
            predictions.append(pred_val)
            targets.append(target_score.item())

    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # --- Calculate Core Benchmark Metrics ---
    # 1. Linear Regression for R^2 and p-value
    slope, intercept, r_value, p_value, std_err = linregress(predictions, targets)
    r_squared = r_value ** 2
    
    # 2. Rank Correlations
    spearman_rho, _ = spearmanr(predictions, targets)
    pearson_r, _ = pearsonr(predictions, targets)

    print("-" * 65)
    print("BENCHMARK METRICS RESULTS:")
    print(f"  • R² (Coefficient of Determination) : {r_squared:.4f}  (Target: ≥ 0.75)")
    print(f"  • p-value                           : {p_value:.4e}  (Target: < 0.05)")
    print(f"  • Spearman Correlation (ρ)         : {spearman_rho:.4f}")
    print(f"  • Pearson Correlation (r)          : {pearson_r:.4f}")
    print("-" * 65)
    
    # Acceptance Validation
    if r_squared >= 0.75 and p_value < 0.05:
        print("SUCCESS: Benchmark targets satisfied (R² ≥ 0.75, p < 0.05).")
    else:
        print("NOTICE: Model requires scaling or further tuning to hit benchmark thresholds.")

if __name__ == "__main__":
    evaluate_phase2_benchmark()