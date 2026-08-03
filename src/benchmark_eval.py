import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from src.dataset import PETaseMutationDataset
from src.model import PETaseStabilityEGNN

def evaluate_phase2_benchmark(pdb_path="data/6eqe.pdb", benchmark_csv="data/benchmark_25.csv", 
                              checkpoint_path="checkpoints/best_egnn.pt"):
    """
    Evaluates the trained E(3)-EGNN checkpoint against Phase 2 ground-truth literature variants.
    Enforces augment_inverse=False to maintain exact 25-row length assertions.
    """
    print("=" * 65)
    print("PHASE 2: GROUND-TRUTH HISTORICAL BENCHMARK EVALUATION")
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
    
    print(f"Loaded {expected_length} benchmark rows from '{benchmark_csv}'")
    
    # Instantiate dataset with augment_inverse=False so length remains exactly 25
    dataset = PETaseMutationDataset(pdb_path=pdb_path, csv_path=benchmark_csv, augment_inverse=False)
    
    # Data-Length Test Assertion
    assert len(dataset) == expected_length, f"[DATA-LENGTH TEST FAILED] Expected {expected_length} rows, but dataset yielded {len(dataset)}."
    print("[DATA-LENGTH TEST PASSED] Verification: Prediction array length matches benchmark dataset.")
    
    # Load Model Checkpoint with 8D input features and 10Å spatial radius
    model = PETaseStabilityEGNN(num_amino_acids=8, emb_dim=32, radius=10.0)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph_data, target_score, mutation_pos, _ = dataset[idx]
            
            # Forward pass through network with 10Å neighborhood pooling
            pred_tensor = model(graph_data, mutation_pos)
            pred_val = pred_tensor.item() if pred_tensor.dim() == 0 else pred_tensor.squeeze().item()
            
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
    
    if np.isnan(spearman_rho): spearman_rho = 0.0
    if np.isnan(pearson_r): pearson_r = 0.0
    if np.isnan(r_squared): r_squared = 0.0
    if np.isnan(p_value): p_value = 1.0
    
    print("-" * 65)
    print("BENCHMARK METRICS RESULTS:")
    print(f"  • R² (Coefficient of Determination) : {r_squared:.4f} (Target: ≥ 0.75)")
    print(f"  • p-value                           : {p_value:.4e} (Target: < 0.05)")
    print(f"  • Spearman Correlation (ρ)         : {spearman_rho:.4f}")
    print(f"  • Pearson Correlation (r)          : {pearson_r:.4f}")
    print("-" * 65)
    
    # Acceptance Validation
    if r_squared >= 0.75 and p_value < 0.05:
        print(" SUCCESS: Phase 2 Benchmark targets satisfied (R² ≥ 0.75, p < 0.05).")
    else:
        print(" NOTICE: Model requires scaling or further tuning to hit benchmark thresholds.")

if __name__ == "__main__":
    evaluate_phase2_benchmark()