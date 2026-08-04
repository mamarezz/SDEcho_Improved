import numpy as np
import pandas as pd
import sys

# Import the actual pipeline components
from src.evaluation import generate_synthetic_dataset
from src.reweighting import compute_gap_decomposition, identify_buckets_for_reweighting
from src.sdecho import sequence_distance, run_sdecho
from src.predicates import Predicate
from src.sequence_builder import build_sequence

def run_toy_example():
    print("=== STEP 1: TOY EXAMPLE VALIDATION ===")
    # Setup: 2 cells, 1 bucket
    # Group A (source, n=10): 8 tuples {covariate=India, outcome=$40,000}, 2 tuples {covariate=USA, outcome=$80,000}
    # Group B (target, n=10): 2 tuples {covariate=India, outcome=$45,000}, 8 tuples {covariate=USA, outcome=$85,000}
    
    data_A = {
        'bucket': [0]*10,
        'covariate': ['India']*8 + ['USA']*2,
        'outcome': [40000]*8 + [80000]*2
    }
    data_B = {
        'bucket': [0]*10,
        'covariate': ['India']*2 + ['USA']*8,
        'outcome': [45000]*2 + [85000]*8
    }
    
    df_A = pd.DataFrame(data_A)
    df_B = pd.DataFrame(data_B)
    
    index = ["0"]
    group_col = "bucket"
    measure_col = "outcome"
    
    # 1. Manual Trace (Expected values)
    # s_source_orig: mean of [40k*8, 80k*2] / 10 = (320k + 160k)/10 = 48,000
    # s_target: mean of [45k*2, 85k*8] / 10 = (90k + 680k)/10 = 77,000
    # d_orig: |77000 - 48000| = 29,000
    
    # 2. Pipeline compute
    # Predicate: Country=USA (to simulate switching from India to USA)
    predicate = Predicate({"covariate": "USA"})
    
    # compute_gap_decomposition returns a GapDecompositionResult object
    res = compute_gap_decomposition(
        df_A, df_B, predicate, group_col, measure_col, index,
        min_cell_support=1 # Allow small sample for toy
    )
    
    print(f"Observed EF: {res.explained_fraction:.4f}")
    print(f"Observed d_orig: {res.d_orig}")
    print(f"Observed d_cf: {res.d_cf}")
    
    if abs(res.explained_fraction - 0.8275) < 0.01:
        print("SUCCESS: Toy example matches hand-computed values.")
    else:
        print("FAILURE: Toy example mismatch!")

def run_synthetic_validation():
    print("\n=== STEP 2: SYNTHETIC VALIDATION AUDIT ===")
    effect_sizes = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for es in effect_sizes:
        # We need to use the actual generator from the code
        df_A, df_B, ground_truth = generate_synthetic_dataset(es, 1000, 42)
        
        # Manually apply predicate (assuming covariate is the one)
        # In the generator, covariate is 0 or 1.
        predicate = Predicate({"covariate": 1})
        
        # We use group_col='bucket' and measure_col='outcome' 
        # (Note: generate_synthetic_dataset creates 'bucket' and 'outcome')
        res = compute_gap_decomposition(
            df_A, df_B, predicate, 'bucket', 'outcome', [0, 1],
            min_cell_support=5
        )
        
        results.append({
            "effect_size": es,
            "ground_truth": ground_truth,
            "estimated": res.explained_fraction,
            "error": res.explained_fraction - ground_truth
        })
    
    print(f"{'Effect Size':<15} | {'GT':<10} | {'Est':<10} | {'Error':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['effect_size']:<15.2f} | {r['ground_truth']:<10.2f} | {r['estimated']:<10.2f} | {r['error']:<10.2f}")

if __name__ == "__main__":
    run_toy_example()
    run_synthetic_validation()
