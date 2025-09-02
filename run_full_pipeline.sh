#!/bin/bash
# Complete BICEP -> ENN-C++ -> FusionAlpha Pipeline Runner
# Demonstrates the full integration on parity task

set -e

echo "🚀 BICEP -> ENN-C++ -> FusionAlpha Complete Pipeline"
echo "===================================================="

# Step 1: Generate BICEP trajectories
echo "Step 1: Generating BICEP parity trajectories..."
cd ./BICEPsrc/BICEPrust/bicep
./target/release/parity_trajectories --sequences 200 --seq-len 15
echo "✅ BICEP trajectories generated"

# Step 2: Convert to CSV for ENN-C++
echo "Step 2: Converting to CSV format..."
python3 -c "
import polars as pl
df = pl.read_parquet('runs/parity_trajectories.parquet')
df_flat = df.with_columns([
    pl.col('state').list.get(0).alias('state_0')
]).drop('state')
df_flat.write_csv('runs/parity_trajectories.csv')
print('✅ Converted to CSV format')
"

# Step 3: Run ENN-C++ training
echo "Step 3: Training ENN-C++ on BICEP trajectories..."
cd ../../../enn-cpp
./apps/bicep_to_enn ../BICEPsrc/BICEPrust/bicep/runs/parity_trajectories.csv
echo "✅ ENN-C++ training completed"

# Step 4: Run FusionAlpha integration
echo "Step 4: Running FusionAlpha with ENN priors..."
cd ../
python3 pipeline_demo.py
echo "✅ FusionAlpha integration completed"

echo ""
echo "🎯 Pipeline Summary:"
echo "  • BICEP: Generated stochastic trajectories for parity task"
echo "  • ENN-C++: Learned sequence patterns with entangled neural networks"
echo "  • FusionAlpha: Used ENN predictions as committor priors for planning"
echo ""
echo "📁 Output files:"
echo "  • BICEP data: BICEPsrc/BICEPrust/bicep/runs/parity_trajectories.parquet"
echo "  • ENN predictions: enn-cpp/enn_predictions.csv"  
echo "  • FusionAlpha graph: FusionAlpha/fusion_graph.json"
echo ""
echo "🚀 Complete pipeline executed successfully!"