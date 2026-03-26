# Adaptive Asynchronous Hierarchical dFLMoE

**Fully decentralized, truly asynchronous federated learning with dynamic Mixture-of-Experts routing for robust training under non-IID data distributions.**

---

## Overview

Standard federated learning relies on a central server, synchronous round barriers, and full-model sharing — all of which break down under heterogeneous data and network conditions. This framework removes all three constraints simultaneously.

Each client maintains a **private body encoder** (never leaves the device) and a lightweight **expert head** (shared with peers over real TCP sockets). A **learned router** selects the top-K most relevant peer experts per input sample using a composite scoring formula that combines trust, feature similarity, staleness decay, and a learned gating signal. Clients organize into **hierarchical clusters** via K-Means on L2-normalized features, reducing communication from O(N²) to O(N√N). Training is **truly asynchronous**: each client runs in its own thread with no global round barrier, so fast clients never wait for slow ones.

The closest prior work is dFLMoE (Xie et al., CVPR 2025), which introduced decentralized head sharing with FSTs. This framework extends that baseline with true asynchrony, composite trust-staleness routing, hierarchical clustering, and a staleness-aware expert cache — addressing the limitations documented in [README_DEEP.md](README_DEEP.md).

---

## Key Features

- **No central server**: fully decentralized P2P communication via real TCP sockets
- **True asynchronous training**: independent per-client threads, no synchronous round barrier; 30% faster than synchronous equivalent with 98.5% of the accuracy
- **Composite expert scoring**: Score = Learned Gate x Trust x Cosine Similarity x Staleness Decay — four-factor per-sample routing that no prior FL framework achieves
- **Head-only sharing**: only the expert head (~134K parameters, ~4% of the full 3.38M client model) is shared — a 96% parameter reduction over full-model FL approaches
- **Hierarchical adaptive clustering**: K-Means on representative features, intra-cluster sharing every round, cross-cluster relay every 60s; 2.61x communication reduction at N=20
- **Staleness-aware expert cache**: automatic eviction at 300s, keep-alive re-sharing for post-training persistence
- **Feature Space Transforms (FSTs)**: identity-initialized learnable alignment layers per peer expert, preventing feature-space mismatch without the FST parameters being frozen (enforced via `add_param_group`)
- **Gradient isolation**: `features.detach()` before the MoE path prevents foreign-expert gradients from corrupting the private body encoder

---

## Architecture

```
ORCHESTRATION (main.py)
  Eval every 300s | Recluster every 150s | eval_pause Event
  |
  +-- Client 0 (Thread)   +-- Client 1 (Thread)   ...   Client N-1 (Thread)
      |                       |
      [Body Encoder]          [Body Encoder]        <- PRIVATE, never shared
      [Expert Head]  <------> [Expert Head]         <- SHARED over TCP
      [Router + FSTs]         [Router + FSTs]       <- local, trains on MoE loss
      [Peer Cache]            [Peer Cache]          <- staleness-evicting store
      |                       |
      +-- TCP Transport (real P2P sockets, length-prefixed framing)

Hierarchical clusters (K=3 default):
  Intra-cluster:  every round  (~10-20s)  knowledge-driven
  Cross-cluster:  every 60s              time-normalized (head-to-head relay)
  Reclustering:   every 150s             structural adaptation
  Eviction:       after 300s             disconnect detection
```

The per-client forward pass applies a hybrid loss:
**L = alpha * L_local + (1 - alpha) * L_MoE** with quadratic warmup from alpha=1.0 to alpha=0.5 over the first 10 rounds.

For full architectural detail, mathematical derivations, and design rationale see [README_DEEP.md](README_DEEP.md).

---

## Results Summary

All experiments use CIFAR-10, 10 clients (unless noted), Dirichlet partitioning, 20 rounds, batch size 64, single seed (42). For full verified analysis with per-evaluation convergence data, see [README_RESULTS_ANALYSIS.md](README_RESULTS_ANALYSIS.md).

### Accuracy vs. Data Heterogeneity (alpha sweep)

| Dirichlet alpha | Heterogeneity | Test Accuracy |
|-----------------|---------------|---------------|
| 0.1             | Extreme       | 63.09%        |
| 0.2             | High          | 68.81%        |
| 0.3             | Moderate      | 73.40%        |
| 0.4             | Mild          | 76.66%        |
| 0.5             | Mild          | 76.82%        |
| IID             | None          | 80.24%        |

### Component Ablation (alpha = 0.3)

| Configuration          | Test Accuracy | Delta vs. Full |
|------------------------|---------------|----------------|
| Full framework (async) | 73.40%        | --             |
| Without MoE routing    | 66.29%        | -7.11 pp       |
| Without warmup         | 72.00%        | -1.40 pp       |
| Without hierarchy      | 72.35%        | -1.05 pp       |
| Synchronous mode       | 74.52%        | +1.12 pp       |

### Asynchronous vs. Synchronous (alpha = 0.3, 20 rounds)

| Mode         | Test Accuracy | Wall-Clock Time | Speedup |
|--------------|---------------|-----------------|---------|
| Synchronous  | 74.52%        | 163.4 min       | 1.0x    |
| Asynchronous | 73.40%        | 113.9 min       | 1.43x   |

Async retains 98.5% of synchronous accuracy (73.40/74.52) while training 30.3% faster. Measured on single-device simulation; real multi-device deployment advantage is expected to be larger due to heterogeneous hardware.

### Scalability (alpha = 0.3, alpha_dirichlet = 0.3)

| N clients | Test Accuracy | Comm. Cost vs. FedAvg |
|-----------|---------------|-----------------------|
| 10 (K=3)  | 73.40%        | 2.98x reduction*      |
| 20 (K=4)  | 68.51%        | 2.61x reduction*      |

*Communication reduction is measured as hierarchical vs flat (all-to-all) total messages. N=20 used different cross-cluster interval (60s vs 300s) and max expert age (300s vs 600s); see [README_RESULTS_ANALYSIS.md](README_RESULTS_ANALYSIS.md) for full analysis.

### Fault Tolerance (alpha = 0.3, N=10)

| Scenario                     | Test Accuracy | Delta    |
|------------------------------|---------------|----------|
| Baseline (no faults)         | 73.40%        | --       |
| 2-client dropout (20%)       | 71.25%        | -2.15 pp |
| Client churn (join/leave)    | 73.62%        | +0.22 pp |
| Majority failure (50%)       | 59.34%        | -14.06 pp|
| Disconnect-rejoin (120s)     | 72.66%        | -0.74 pp |

Graceful degradation under dropout; churn is handled transparently by the staleness cache.

---

## Quick Start

### Requirements

```
Python >= 3.8
PyTorch >= 1.10
scikit-learn
numpy
```

### Installation

```bash
git clone https://github.com/<your-org>/Adaptive-Asynchronous-Hierarchical-dFLMoE.git
cd Adaptive-Asynchronous-Hierarchical-dFLMoE
pip install torch torchvision scikit-learn numpy
```

### Run (default: CIFAR-10, 10 clients, alpha=0.5, 20 rounds)

```bash
python3 main.py \
    --dataset cifar10 \
    --num_clients 10 \
    --num_clusters 3 \
    --rounds 20 \
    --local_epochs 3 \
    --partition_method dirichlet \
    --non_iid_alpha 0.5 \
    --alpha 0.5 \
    --warmup_rounds 10 \
    --top_k_experts 3 \
    --staleness_lambda 0.005 \
    --max_expert_age 300 \
    --cross_cluster_interval 60 \
    --eval_interval 300 \
    --staleness_floor 0.1 \
    --recluster_interval 150 \
    --lr_head 0.001 \
    --lr_body 0.001 \
    --lr_router 0.001 \
    --weight_decay 1e-4 \
    --lr_decay 0.98 \
    --dropout 0.3 \
    --batch_size 64 \
    --seed 42
```

Key parameters:

| Parameter               | Default | Description                                            |
|-------------------------|---------|--------------------------------------------------------|
| `--non_iid_alpha`       | 0.5     | Dirichlet concentration (lower = more heterogeneous)   |
| `--alpha`               | 0.5     | Local loss weight (1-alpha goes to MoE)                |
| `--top_k_experts`       | 3       | Experts selected per sample                            |
| `--staleness_lambda`    | 0.005   | Staleness decay rate lambda                            |
| `--max_expert_age`      | 300     | Expert cache eviction threshold (seconds)              |
| `--cross_cluster_interval` | 60   | Cross-cluster exchange interval (seconds)              |
| `--staleness_floor`     | 0.1     | Minimum staleness factor (prevents expert suppression) |
| `--warmup_rounds`       | 10      | Rounds for quadratic alpha warmup                      |

---

## Project Structure

```
Adaptive-Asynchronous-Hierarchical-dFLMoE/
├── main.py                  # Orchestration: data loading, async training, evaluation
├── client_node.py           # Client: training loop, hybrid loss, expert sharing
├── models/
│   ├── body_encoder.py      # SimpleCNNBody (private, 3.2M params, 32x32 input)
│   ├── head.py              # Expert Head (shared, 134K params, MLP classifier)
│   ├── fst.py               # Feature Space Transform (identity-init, 512x512)
│   └── router.py            # Composite scoring + top-K MoE aggregation
├── infra/
│   ├── peer_cache.py        # Thread-safe expert cache, staleness eviction
│   ├── transport.py         # TCP P2P transport, length-prefixed framing
│   └── cluster.py           # K-Means hierarchical clustering
├── utils/
│   └── data_utils.py        # Dirichlet / label-sharding / IID partitioning
├── results/                 # Experiment outputs organized by dataset and alpha
├── README_DEEP.md           # Full technical reference: math, design decisions, fixes
└── README.md                # This file
```

---

## Citation

If you use this framework in your work, please cite:

```bibtex
@misc{dflmoe_async_hierarchical_2025,
  title   = {Adaptive Asynchronous Hierarchical Decentralized Federated Learning
             with Dynamic Mixture of Experts},
  year    = {2025},
  note    = {https://github.com/<your-org>/Adaptive-Asynchronous-Hierarchical-dFLMoE}
}
```

---

## References

1. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
2. Li, T., et al. "Federated Optimization in Heterogeneous Networks (FedProx)." MLSys 2020.
3. Nguyen, J., et al. "Federated Learning with Buffered Asynchronous Aggregation." AISTATS 2022.
4. Xie, Z., et al. "dFLMoE: Decentralized Federated Learning via Mixture of Experts." CVPR 2025.

---

For deep technical documentation including the full scoring formula derivation, gradient flow analysis, all 14 design iteration notes, and per-module implementation details, see [README_DEEP.md](README_DEEP.md).
