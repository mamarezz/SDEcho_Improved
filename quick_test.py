"""
Quick smoke test to verify imports and basic functionality.
"""
import sys
sys.path.insert(0, 'src')

print("Testing imports...")

try:
    from src.predicates import Predicate, enumerate_predicates, predicate_mask
    print("✓ predicates.py imports successfully")
    
    from src.sequence_builder import build_sequence
    print("✓ sequence_builder.py imports successfully")
    
    from src.sdecho import SDEchoResult, sequence_distance, run_sdecho
    print("✓ sdecho.py imports successfully")
    
    from src.reweighting import (
        ReweightingDiagnostics, GapDecompositionResult,
        select_predicate, compute_cell_weights,
        weighted_aggregate_sequence, compute_gap_decomposition
    )
    print("✓ reweighting.py imports successfully")
    
    from src.evaluation import (
        removal_baseline, bootstrap_explained_fraction_ci,
        generate_synthetic_dataset, run_ablation
    )
    print("✓ evaluation.py imports successfully")
    
    from src.data_loader import load_and_preprocess_data, split_groups, get_bucket_index
    print("✓ data_loader.py imports successfully")
    
    from src.visualization import (
        plot_sequence_comparison, plot_gap_decomposition_bar,
        plot_weight_distribution, render_diagnostics_table
    )
    print("✓ visualization.py imports successfully")
    
    print("\n" + "="*60)
    print("ALL IMPORTS SUCCESSFUL ✓")
    print("="*60)
    
    # Quick functional test
    print("\nQuick functional test...")
    
    import pandas as pd
    import numpy as np
    
    # Test Predicate
    p = Predicate({"Country": "USA"})
    print(f"✓ Predicate created: {p}")
    
    # Test build_sequence
    df = pd.DataFrame({
        "bucket": ["0-2", "0-2", "3-5"],
        "value": [100.0, 200.0, 300.0]
    })
    seq = build_sequence(df, "bucket", "value", "mean", ["0-2", "3-5"])
    print(f"✓ Sequence built: {seq}")
    
    # Test sequence_distance
    s1 = np.array([1.0, 2.0])
    s2 = np.array([1.0, 3.0])
    dist = sequence_distance(s1, s2)
    print(f"✓ Distance computed: {dist:.2f}")
    
    print("\n" + "="*60)
    print("QUICK TEST PASSED ✓")
    print("="*60)
    print("\nAll modules are working correctly!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)