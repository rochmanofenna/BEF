# BICEP → ENN-C++ → FusionAlpha Pipeline

A complete research pipeline integrating three cutting-edge components for stochastic trajectory generation, entangled neural network learning, and committor-based planning.

## 🚀 **Overview**

This pipeline demonstrates the complete integration of:

- **BICEP**: Brownian-Inspired Computation Engine for Paths (Rust) - SDE trajectory generation
- **ENN-C++**: Entangled Neural Networks (C++) - High-performance sequence learning with BPTT
- **FusionAlpha**: Committor-based planning system (Rust/Python) - Graph reasoning with severity scaling

## 📊 **Performance Results**

| Component | Input | Output | Performance |
|-----------|-------|--------|-------------|
| BICEP | Random seeds | 200 parity trajectories | <1s generation |
| ENN-C++ | BICEP trajectories | Committor predictions | 45% → 85% accuracy |
| FusionAlpha | ENN predictions | Planning decisions | 85% → 100% accuracy |

**Total Pipeline Runtime**: ~15 seconds  
**Final Accuracy**: 100% on training data, ~49% on new sequences

## 🏗️ **Architecture**

```
BICEP (Rust)     →     ENN-C++ (C++)     →     FusionAlpha (Rust/Python)
├─ SDE Integration     ├─ Entangled Cells      ├─ Graph Construction
├─ Trajectory Gen      ├─ BPTT Training        ├─ Severity Scaling  
└─ Parquet Output      └─ Committor Learning   └─ Decision Making
```

## 🛠️ **Quick Start**

### Prerequisites
- Rust (1.70+)
- C++17 compiler (GCC/Clang)  
- Python 3.9+
- Eigen3 (auto-downloaded)

### Run Complete Pipeline
```bash
# Make executable and run
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

This will:
1. Build all components
2. Generate BICEP parity trajectories  
3. Train ENN-C++ on the data
4. Run FusionAlpha planning
5. Display results

### Manual Execution

#### 1. Build BICEP
```bash
cd BICEPsrc/BICEPrust/bicep
cargo build --release
```

#### 2. Generate Trajectories
```bash
./target/release/parity_trajectories --sequences 200 --seq-len 15
```

#### 3. Build ENN-C++
```bash
cd ../../../enn-cpp
make all
```

#### 4. Train ENN
```bash
# Convert parquet to CSV first
python3 -c "import polars as pl; df = pl.read_parquet('../BICEPsrc/BICEPrust/bicep/runs/parity_trajectories.parquet'); df.with_columns([pl.col('state').list.get(0).alias('state_0')]).drop('state').write_csv('parity_data.csv')"

# Run ENN training
./apps/bicep_to_enn parity_data.csv
```

#### 5. Run FusionAlpha
```bash
cd ../
python3 pipeline_demo.py
```

## 📁 **Repository Structure**

```
bicep-enn-fusion-clean/
├── BICEPsrc/                    # Rust-based SDE trajectory generator
│   └── BICEPrust/bicep/
│       ├── crates/              # BICEP core modules
│       └── target/release/      # Compiled binaries
├── enn-cpp/                     # High-performance ENN implementation
│   ├── include/enn/             # Header files
│   ├── src/                     # Implementation
│   ├── apps/                    # Applications
│   ├── tests/                   # Validation tests
│   └── Makefile                 # Build system
├── FusionAlpha/                 # Committor-based planning
│   ├── crates/                  # Rust core
│   ├── python/                  # Python bindings
│   └── examples/                # Demo scripts
├── run_full_pipeline.sh         # Complete pipeline runner
├── pipeline_demo.py             # Python integration demo
└── FUSION_ALPHA_USAGE_GUIDE.md  # Detailed usage instructions
```

## 🎯 **Key Features**

### BICEP
- **SDE Integration**: Euler-Maruyama, Heun, Milstein methods
- **Parity Task**: Custom trajectory generator for binary sequences
- **Parquet Output**: Efficient data serialization

### ENN-C++
- **Entangled Cells**: ψₜ₊₁ = tanh(Wₓxₜ + Wₕhₜ + (E-λI)ψₜ + b)
- **BPTT**: Full backpropagation through time
- **OpenMP**: Parallel batch processing
- **Validation**: Comprehensive gradient checks (1e-10 precision)

### FusionAlpha  
- **Committor Functions**: Transition probability learning
- **Severity Scaling**: Confidence-weighted graph propagation
- **Active Learning**: Online improvement with feedback

## 📊 **Technical Specifications**

| Metric | BICEP | ENN-C++ | FusionAlpha |
|--------|-------|---------|-------------|
| Build Time | 3s | 2s | <1s |
| Memory Usage | ~50MB | ~100MB | ~20MB |
| Parallelization | Single-threaded | OpenMP | Graph-parallel |
| Data Format | Parquet | CSV/Memory | JSON |

## 🔬 **Research Applications**

- **Molecular Dynamics**: Rare event prediction
- **Financial Modeling**: Option pricing with uncertainty
- **Reinforcement Learning**: Goal-conditioned planning
- **Time Series**: Sequential decision making

## 📈 **Performance Notes**

- Designed for research scale (200-1000 sequences)
- Production scale requires HPC/A100 resources
- Parity task difficulty scales as O(2^N) for N-bit sequences
- Current implementation optimized for understanding over scale

## 🤝 **Contributing**

This is a research demonstration pipeline. For production use:
1. Scale BICEP to generate 100K+ trajectories
2. Add GPU acceleration to ENN-C++
3. Implement distributed FusionAlpha graph processing

## 📄 **License**

Research and educational use. Components based on established mathematical foundations in stochastic processes, neural networks, and graph theory.

---

**Built for high-performance scientific computing and research applications** 🚀