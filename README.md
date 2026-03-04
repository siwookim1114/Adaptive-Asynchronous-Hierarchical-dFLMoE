# Adaptive Asynchronous Hierarchical Decentralized Federated Learning with Dynamic Mixture of Experts (dFLMoE)

> A fully decentralized, truly asynchronous federated learning framework that combines hierarchical peer-to-peer communication with dynamic Mixture-of-Experts routing for robust learning under non-IID data distributions.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Key Contributions](#2-key-contributions)
3. [System Architecture](#3-system-architecture)
4. [Core Modules](#4-core-modules)
   - [4.1 Body Encoder (Private Feature Extractor)](#41-body-encoder-private-feature-extractor)
   - [4.2 Expert Head (Shared Classifier)](#42-expert-head-shared-classifier)
   - [4.3 Feature Space Transform (FST)](#43-feature-space-transform-fst)
   - [4.4 Router (Dynamic Expert Routing)](#44-router-dynamic-expert-routing)
   - [4.5 Peer Cache (Staleness-Aware Storage)](#45-peer-cache-staleness-aware-storage)
   - [4.6 Transport Layer (TCP P2P)](#46-transport-layer-tcp-p2p)
   - [4.7 Cluster Manager (Hierarchical Organization)](#47-cluster-manager-hierarchical-organization)
5. [Routing & Scoring: The Heart of the System](#5-routing--scoring-the-heart-of-the-system)
6. [End-to-End Data Flow](#6-end-to-end-data-flow)
7. [Communication Protocol](#7-communication-protocol)
8. [True Asynchronous Training](#8-true-asynchronous-training)
9. [Training Pipeline](#9-training-pipeline)
10. [Data Partitioning](#10-data-partitioning)
11. [Global Evaluation](#11-global-evaluation)
12. [Design Evolution: Trials, Errors & Fixes](#12-design-evolution-trials-errors--fixes)
13. [Comparison with Prior Work](#13-comparison-with-prior-work)
14. [Experiment Results](#14-experiment-results)
15. [Setup & Usage](#15-setup--usage)
16. [References](#16-references)

---

## 1. Abstract

Standard federated learning (FL) approaches such as FedAvg rely on a **central server** for model aggregation, operate in **synchronous rounds** (where the slowest client becomes the bottleneck), and share **full model parameters** (exposing private representations). Under non-IID data distributions, these methods suffer from severe **client drift** where locally trained models diverge.

This framework addresses all four limitations simultaneously:

| Problem | Standard FL | Our Solution |
|---------|-------------|--------------|
| Central server bottleneck | Server aggregates all updates | **Fully decentralized** P2P with hierarchical clustering |
| Synchronous round barrier | Fastest client waits for slowest | **True asynchronous** training (no round barrier) |
| Full model sharing | Entire model transits network | **Head-only sharing** (~134K vs ~3.2M params, 96% reduction) |
| Client drift under non-IID | Uniform weighted averaging | **Dynamic MoE routing** with trust, similarity, staleness scoring |

Each client maintains a **private body encoder** (never shared) and a lightweight **expert head** (shared with peers). A **learned router** dynamically selects and weights the top-K most relevant experts for each input sample using a composite scoring formula that incorporates trust, feature similarity, staleness decay, and learned gating attention.

---

## 2. Key Contributions

1. **True Asynchronous Decentralized FL**: Each client trains independently in its own thread with no global round synchronization. Expert packages arrive and are consumed asynchronously via real TCP sockets. Communication triggers use a hybrid strategy: knowledge-driven (round-based) for intra-cluster and resource-driven (wall-clock time-based) for cross-cluster.

2. **Composite Expert Scoring Formula**: Unlike prior work that uses uniform averaging (FedAvg) or simple attention (dFLMoE), our router computes:

$$\text{Score}_{ij} = \underbrace{\sigma\!\left(\frac{\text{proj}(f_i) \cdot \text{emb}_j}{\sqrt{d}}\right)}_{\text{Learned Gating}} \times \underbrace{T_j}_{\text{Trust}} \times \underbrace{S_{ij}}_{\text{Similarity}} \times \underbrace{e^{-\lambda \Delta t_j}}_{\text{Staleness}}$$

   This four-factor score enables **per-sample, per-expert** dynamic routing — a level of granularity no prior FL framework achieves.

3. **Hierarchical Adaptive Clustering**: K-Means clustering on L2-normalized feature vectors organizes clients into clusters. Intra-cluster sharing is frequent (every round); cross-cluster sharing is mediated through cluster heads on a wall-clock timer, reducing communication from $O(N^2)$ to $O(N)$.

4. **Feature Space Transforms (FSTs)**: Identity-initialized linear transforms that learn to align features across clients with heterogeneous data distributions, enabling meaningful expert head reuse.

5. **Staleness-Aware Expert Cache**: Automatic eviction of stale experts (default 300s), keep-alive re-sharing for post-training persistence, and dynamic cache fluctuation reflecting genuine connectivity.

---

## 3. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION (main.py)                      │
│  Time-based evaluation (120s) │ Time-based reclustering (150s)      │
│  eval_pause Event             │ stats_queue monitoring              │
└────────────┬────────────┬────────────┬────────────┬─────────────────┘
             │            │            │            │
    ┌────────▼───┐ ┌──────▼─────┐ ┌───▼────────┐ ┌▼────────────┐
    │ Client 0   │ │ Client 1   │ │ Client 2   │ │ Client N-1  │
    │ (Thread)   │ │ (Thread)   │ │ (Thread)   │ │ (Thread)    │
    │            │ │            │ │            │ │             │
    │ ┌────────┐ │ │ ┌────────┐ │ │ ┌────────┐ │ │ ┌─────────┐ │
    │ │  Body  │ │ │ │  Body  │ │ │ │  Body  │ │ │ │  Body   │ │
    │ │Encoder │ │ │ │Encoder │ │ │ │Encoder │ │ │ │ Encoder │ │
    │ │PRIVATE │ │ │ │PRIVATE │ │ │ │PRIVATE │ │ │ │ PRIVATE │ │
    │ └───┬────┘ │ │ └───┬────┘ │ │ └───┬────┘ │ │ └────┬────┘ │
    │     │      │ │     │      │ │     │      │ │      │      │
    │ ┌───▼────┐ │ │ ┌───▼────┐ │ │ ┌───▼────┐ │ │ ┌────▼────┐ │
    │ │  Head  │ │ │ │  Head  │ │ │ │  Head  │ │ │ │  Head   │ │
    │ │SHARED  │◄├─┼─┤SHARED  │◄├─┼─┤SHARED  │◄├─┼─┤ SHARED  │ │
    │ └────────┘ │ │ └────────┘ │ │ └────────┘ │ │ └─────────┘ │
    │ ┌────────┐ │ │ ┌────────┐ │ │ ┌────────┐ │ │ ┌─────────┐ │
    │ │ Router │ │ │ │ Router │ │ │ │ Router │ │ │ │ Router  │ │
    │ │+FSTs   │ │ │ │+FSTs   │ │ │ │+FSTs   │ │ │ │ +FSTs   │ │
    │ └────────┘ │ │ └────────┘ │ │ └────────┘ │ │ └─────────┘ │
    │ ┌────────┐ │ │ ┌────────┐ │ │ ┌────────┐ │ │ ┌─────────┐ │
    │ │  Peer  │ │ │ │  Peer  │ │ │ │  Peer  │ │ │ │  Peer   │ │
    │ │ Cache  │ │ │ │ Cache  │ │ │ │ Cache  │ │ │ │  Cache  │ │
    │ └────────┘ │ │ └────────┘ │ │ └────────┘ │ │ └─────────┘ │
    └──────┬─────┘ └──────┬─────┘ └──────┬─────┘ └──────┬──────┘
           │              │              │               │
           └──────────────┼──────────────┼───────────────┘
                          │              │
                    TCP Transport (Real P2P Sockets)
```

### Per-Client Component Ownership

| Component | Parameters | Shared? | Trainable? |
|-----------|-----------|---------|------------|
| **Body Encoder** (SimpleCNNBody) | 3,244,864 | No (private) | Yes |
| **Expert Head** (MLP) | 133,898 | **Yes** (exchanged) | Yes |
| **Router** (projector + embeddings) | 133,888 | No | Yes |
| **FSTs** (up to 10 Linear transforms) | 262,656 each | No | Yes |
| **Total per client** | ~6.1M | | |
| **Shared over network** | **133,898 (2.2%)** | | |

---

## 4. Core Modules

### 4.1 Body Encoder (Private Feature Extractor)

**File**: `models/body_encoder.py`

The body encoder is each client's **private** feature extractor — it never leaves the client. This preserves data privacy: raw input representations stay local, and only the lightweight classifier head (which operates on abstract features) is shared.

#### Architecture: SimpleCNNBody

A VGG-style CNN with three convolutional blocks, each containing two convolutions with batch normalization, ReLU activation, and 2x spatial downsampling via max pooling:

```
Input: (B, C, 32, 32)    C=3 for CIFAR-10, C=1 for MNIST

Block 1: Conv(C→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU → MaxPool(2×2)
         (B, C, 32, 32) → (B, 64, 16, 16)

Block 2: Conv(64→128, 3×3) → BN → ReLU → Conv(128→128, 3×3) → BN → ReLU → MaxPool(2×2)
         (B, 64, 16, 16) → (B, 128, 8, 8)

Block 3: Conv(128→256, 3×3) → BN → ReLU → Conv(256→256, 3×3) → BN → ReLU → MaxPool(2×2)
         (B, 128, 8, 8) → (B, 256, 4, 4)

Projection: Flatten → Linear(4096→512) → ReLU
            (B, 256, 4, 4) → (B, 4096) → (B, 512)

Output: f ∈ R^{B×512}    (feature vectors)
```

**Parameter count (CIFAR-10):** 3,244,864

**Design rationale**: Three blocks with doubling channels (64→128→256) and halving spatial dimensions (32→16→8→4) provide sufficient receptive field for 32×32 inputs. The final linear projection compresses the 4096-dim flattened feature map into a compact 512-dim space suitable for efficient similarity computation and expert routing.

---

### 4.2 Expert Head (Shared Classifier)

**File**: `models/head.py`

The expert head is the unit of exchange between clients — the only component that transits the network. It is intentionally lightweight to minimize communication cost.

```
Input:  f ∈ R^{B×512}      (features from body encoder)

Linear(512→256) → ReLU → Dropout(p=0.3) → Linear(256→10)

Output: logits ∈ R^{B×10}   (raw class scores, pre-softmax)
```

**Parameter count:** 133,898 (~4.1% of body encoder size)

**Communication efficiency**: Sharing only the head means each expert package transmits ~134K parameters instead of ~3.4M (the full model). This is a **96% reduction** in communication cost compared to FedAvg-style full model sharing.

**What gets packaged** (sent over network):

| Field | Type | Purpose |
|-------|------|---------|
| `head_state_dict` | OrderedDict | Head model weights |
| `client_id` | str | Source identifier |
| `timestamp` | float | Creation time (wall-clock) |
| `trust_score` | float | Source's validation-derived trust |
| `validation_accuracy` | float | Source's latest val accuracy |
| `representative_features` | Tensor(512) | Mean feature vector for similarity |
| `num_samples` | int | Training dataset size |

---

### 4.3 Feature Space Transform (FST)

**File**: `models/fst.py`

**The problem**: When Client A sends its expert head to Client B, Client B's body encoder produces features in a *different* feature space than Client A's body encoder (because they trained on different data distributions). Directly feeding Client B's features into Client A's head yields poor predictions.

**The solution**: A learnable linear transform per expert that aligns the local feature space to the remote expert's expected input space.

```
FST_j: R^{512} → R^{512}

FST_j(f) = W_j · f + b_j

Initialization: W_j = I (identity matrix), b_j = 0 (zero vector)
```

**Parameter count per FST:** 262,656 (512×512 + 512)

**Why identity initialization**: At startup, FST(f) = f — the transform is a no-op. This ensures that the local expert (which doesn't need alignment) works perfectly from round 1. As training progresses, FSTs for remote experts gradually learn the necessary rotation/scaling to map local features into each remote expert's feature space.

**Why linear (no nonlinearity)**: The body encoder already produces rich nonlinear features. The FST only needs to perform an affine alignment (rotation, scaling, shifting) between feature spaces — adding nonlinearity would be redundant and harder to optimize.

**Critical implementation detail**: FSTs are created lazily via `get_or_create_fst()` when a new expert is first encountered. Since they are created *after* the router's optimizer is initialized, new FST parameters must be explicitly added via `optimizer.add_param_group()` — otherwise their gradients accumulate but parameters never update.

---

### 4.4 Router (Dynamic Expert Routing)

**File**: `models/router.py`

The router is the central intelligence of the system. It scores each available expert for each input sample using a composite formula that combines metadata-based scoring with learned attention-style gating.

#### Learnable Components

| Component | Shape | Parameters | Purpose |
|-----------|-------|------------|---------|
| `feature_projector` | Linear(512→256) | 131,328 | Projects features for gating attention |
| `expert_embeddings` | Embedding(N, 256) | 2,560 | Learnable identity vector per expert |

**Total router parameters (excluding FSTs):** 133,888

#### Expert Metadata (per registered expert)

| Field | Update Rule | Range |
|-------|-------------|-------|
| Trust score $T_j$ | EMA: $T_j \leftarrow 0.95 \cdot T_j + 0.05 \cdot p$ | [0.1, 1.0] |
| Representative features $f_j$ | EMA: $f_j \leftarrow 0.9 \cdot f_j + 0.1 \cdot f_{new}$ | $\mathbb{R}^{512}$ |
| Last update timestamp | Direct replacement | UNIX timestamp |
| Validation accuracy | Direct replacement | [0, 1] |

#### The Complete Scoring Formula

For input features $f_i$ from client $i$ and expert $j$:

**Step 1 — Trust Score** $T_j$:
$$T_j = \text{clamp}(0.95 \cdot T_j^{old} + 0.05 \cdot \text{val\_acc}_j,\ 0.1,\ 1.0)$$

Trust is the EMA-smoothed validation accuracy of the expert's source client. Higher accuracy → higher trust. Clamped to [0.1, 1.0] so no expert is completely silenced.

**Step 2 — Similarity Score** $S_{ij}$:
$$S_{ij} = \max\!\left(0,\ \frac{f_i \cdot f_j^{repr}}{||f_i|| \cdot ||f_j^{repr}||}\right)$$

Cosine similarity between the input batch features and the expert's representative feature vector, clamped to non-negative. This measures how relevant expert $j$'s training distribution is to the current input.

**Why `max(0, ·)` instead of `(sim+1)/2`**: The common `(sim+1)/2` mapping compresses the useful range. Raw cosines of 0.6–0.9 become 0.8–0.95 (only 15% spread). With `max(0, ·)`, the same cosines stay as 0.6–0.9 (30% spread, **2× more discriminative**). In practice, mean feature vectors between clients on the same dataset are always positively correlated, so negative values are rare.

**Step 3 — Staleness Decay** $e^{-\lambda \Delta t_j}$:
$$\text{Staleness}_j = e^{-\lambda \cdot (t_{now} - t_j^{update})}$$

Exponential decay based on how old the expert's last update is. With $\lambda = 0.005$:

| Expert age | Staleness factor | Interpretation |
|------------|-----------------|----------------|
| 0s (just received) | 1.000 | Perfectly fresh |
| 30s (intra-cluster) | 0.861 | Very fresh |
| 60s (one cross-cluster cycle) | 0.741 | Fresh |
| 150s (stale) | 0.472 | Degraded |
| 300s (eviction threshold) | 0.223 | Nearly evicted |

**Step 4 — Base Score**:
$$\text{Base}_{ij} = T_j \times S_{ij} \times e^{-\lambda \Delta t_j}$$

Multiplicative combination. All three factors are in [0, 1], so the base score is in [0, 1].

**Step 5 — Learned Gating (Scaled Dot-Product Attention)**:
$$g_{ij} = \sigma\!\left(\frac{\text{proj}(f_i) \cdot \text{emb}_j}{\sqrt{256}}\right)$$

Where `proj` is the feature projector (Linear 512→256), `emb_j` is the learnable expert embedding, and $\sigma$ is the sigmoid function.

**Step 6 — Final Score**:
$$\text{Score}_{ij} = g_{ij} \times \text{Base}_{ij}$$

**Step 7 — Top-K Selection and Softmax**:
$$\text{top\_scores}, \text{top\_indices} = \text{topk}(\text{Scores}, K=3)$$
$$w_k = \text{softmax}(\text{top\_scores} / \tau)_k \quad \text{for } k = 1, \ldots, K$$

The top-K experts per sample receive softmax-normalized weights; all others receive zero weight (sparse routing).

**Step 8 — MoE Output**:
$$\hat{y}_{MoE} = \sum_{k=1}^{K} w_k \cdot \text{Head}_k(\text{FST}_k(f_i))$$

#### Why This Design Differs from Standard MoE Attention

| Aspect | Standard Transformer MoE | Our Router |
|--------|--------------------------|------------|
| Scoring | Softmax over all experts (dense) | Per-expert sigmoid × metadata (sparse) |
| Selection | Load-balanced gating (Switch/GShard) | Top-K with trust+staleness pruning |
| Context | Token features only | Features + trust + similarity + temporal |
| Training signal | All experts compete uniformly | Metadata pre-filters; gating fine-tunes |
| Adaptivity | Static once trained | Dynamic — scores change as experts arrive/depart/age |

The key insight: in federated learning, we have **rich metadata** about each expert (trust, similarity, freshness) that standard MoE settings don't have. Our scoring formula injects this domain knowledge directly, making the router's job easier — it only needs to learn the residual gating adjustment, not the entire routing logic from scratch.

---

### 4.5 Peer Cache (Staleness-Aware Storage)

**File**: `infra/peer_cache.py`

Thread-safe storage for expert packages received from peers, with automatic staleness-based eviction.

```
┌─────────────────────────────────────────┐
│              PEER CACHE                  │
│                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │Expert    │ │Expert    │ │Expert    ││
│  │client_0  │ │client_3  │ │client_7  ││
│  │age: 15s  │ │age: 45s  │ │age: 280s ││
│  │trust:0.92│ │trust:0.78│ │trust:0.65││
│  └──────────┘ └──────────┘ └──────────┘│
│                                          │
│  Eviction Policy:                        │
│  1. age > 300s → auto-evict (stale)     │
│  2. cache full → evict oldest            │
│  3. duplicate → keep only if newer       │
│                                          │
│  Thread Safety: RLock on all operations  │
└─────────────────────────────────────────┘
```

**Configuration**:
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_cache_size` | 50 (set to N clients) | Maximum experts in cache |
| `max_age_seconds` | 300.0 | Auto-eviction threshold |
| `staleness_decay` | 0.005 | Lambda for scoring |
| `auto_evict` | True | Evict on every add() |

**Eviction semantics**: Eviction represents a genuinely **unreachable client** (crashed, disconnected). It does NOT mean "finished training" — clients that finish training enter a keep-alive loop that re-shares their expert every 30s, preventing eviction. Only if a client disappears for >300s (no keep-alive received) does its expert get evicted.

---

### 4.6 Transport Layer (TCP P2P)

**File**: `infra/transport.py`

Real TCP socket-based peer-to-peer communication — not a simulation.

#### Message Protocol
```
┌─────────────────────────────────────────┐
│ 4 bytes: message length (uint32, big-endian) │
│ N bytes: pickle(Message object)              │
└─────────────────────────────────────────┘
```

The 4-byte length prefix enables reliable framing over TCP's streaming protocol. `recv_exact()` handles partial reads by looping until exactly N bytes are received.

#### Threading Model (per client)

```
┌─────────────────────────────────────────────────┐
│                   Client i                       │
│                                                  │
│  Training Thread          Server Thread          │
│  ├─ train_round()         ├─ accept_loop()       │
│  ├─ share_expert()        │  └─ accept() → spawn │
│  │  └─ transport.send()   │                      │
│  └─ keep-alive loop       │  Handler Thread 1    │
│                           │  ├─ recv_exact(4)    │
│  transport.send() ───────►│  ├─ recv_exact(N)    │
│  (per-peer send lock)     │  ├─ Message.from_bytes│
│                           │  └─ dispatch_message()│
│                           │     └─ handler(msg)   │
│                           │        (lock-free)    │
│                           │                      │
│                           │  Handler Thread 2    │
│                           │  └─ (same pattern)   │
└─────────────────────────────────────────────────┘
```

#### Thread Safety Design

| Lock | Type | Protects | Held During |
|------|------|----------|-------------|
| `peer_lock` | RLock | Socket dict | Socket lookup/creation only |
| `send_locks[peer_id]` | Lock (per-peer) | Socket writes | Length prefix + message data send |
| `handler_lock` | RLock | Handler dict | Dict lookup only (NOT handler execution) |
| `stats_lock` | Lock | Statistics counters | Counter increment |

Critical design: `handler_lock` is held only during the dictionary lookup (microseconds), NOT during handler execution. This prevents relay broadcasts (which call `send()` internally) from blocking incoming message reception.

---

### 4.7 Cluster Manager (Hierarchical Organization)

**File**: `infra/cluster.py`

Organizes clients into clusters based on feature similarity, reducing communication complexity.

#### K-Means Clustering Algorithm

```
1. Collect representative features: F ∈ R^{N×512}
2. L2-normalize: F_norm[i] = F[i] / max(||F[i]||, 1e-8)
3. K-Means(n_clusters=3, n_init=10) on F_norm
4. Assign clients to clusters
5. Select cluster heads (highest trust per cluster)
```

**Why L2 normalization before K-Means**: Without normalization, Euclidean-distance K-Means in 512 dimensions is dominated by feature *magnitude* differences. L2 normalization converts Euclidean distance to an angular metric (effectively 1 − cosine_similarity), making clustering based on feature *direction* — which reflects the learned data distribution — rather than magnitude.

#### Hierarchical Structure

```
                    ┌──────────────────────┐
                    │   Cross-Cluster      │
                    │   (Head-to-Head)     │
                    │   Time-based: 60s    │
                    └──────┬───────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ Cluster 0  │   │ Cluster 1  │   │ Cluster 2  │
    │            │   │            │   │            │
    │ client_0 ★ │   │ client_3 ★ │   │ client_7 ★ │
    │ client_1   │   │ client_4   │   │ client_8   │
    │ client_2   │   │ client_5   │   │ client_9   │
    │            │   │ client_6   │   │            │
    └────────────┘   └────────────┘   └────────────┘
        ★ = cluster head (highest trust)

    Intra-cluster: Every round (round-based)
    Cross-cluster: Every 60s  (time-based)
    Reclustering:  Every 150s (time-based)
```

#### Communication Reduction

| Topology | Connections per cycle | Reduction |
|----------|----------------------|-----------|
| Full mesh (N=10) | 10 × 9 = 90 | — |
| Hierarchical (K=3) | ~29 | **67.8%** |

---

## 5. Routing & Scoring: The Heart of the System

This section provides the complete mathematical formulation of the routing mechanism — the key differentiator of this framework.

### Complete Scoring Pipeline

```
For each sample x_b in batch B, for each available expert j:

    ┌─────────────────────────────────────────────────────────────┐
    │                    SCORING PIPELINE                          │
    │                                                              │
    │  Trust:     T_j ∈ [0.1, 1.0]   (EMA of val accuracy)       │
    │       ×                                                      │
    │  Similarity: S_ij = max(0, cos(f_b, f_j^repr))              │
    │       ×                                                      │
    │  Staleness: exp(-λ · Δt_j)      (exponential decay)         │
    │       =                                                      │
    │  Base Score: T_j · S_ij · exp(-λ · Δt_j)                    │
    │       ×                                                      │
    │  Gate: σ(proj(f_b) · emb_j / √d)  (learned attention)      │
    │       =                                                      │
    │  Final Score                                                 │
    │                                                              │
    │  ──── Top-K Selection (K=3) ────                             │
    │  ──── Softmax Normalization ────                             │
    │  ──── Weighted Sum of Expert Outputs ────                    │
    └─────────────────────────────────────────────────────────────┘
```

### Why Each Factor Matters

| Factor | Without it | With it | Effect |
|--------|-----------|---------|--------|
| **Trust** $T_j$ | All experts weighted equally regardless of quality | High-accuracy experts dominate | Prevents low-quality experts from corrupting predictions |
| **Similarity** $S_{ij}$ | Expert from unrelated data distribution may be selected | Experts trained on similar data preferred | Ensures selected experts have relevant knowledge |
| **Staleness** $e^{-\lambda \Delta t}$ | Outdated experts weighted same as fresh ones | Fresh experts preferred | Ensures predictions use up-to-date parameters |
| **Gating** $\sigma(\cdot)$ | Fixed scoring, no learning | Router learns input-dependent routing | Captures patterns that metadata cannot express |

### Comparison: Our Routing vs. Dense Attention

**Dense attention** (used in standard Transformer MoE and the original dFLMoE):
$$y = \text{softmax}(Q \cdot K^T / \sqrt{d}) \cdot V$$

All experts participate with softmax-normalized weights. No explicit trust, staleness, or similarity — the model must learn everything from data.

**Our composite routing**:
$$y = \sum_{k \in \text{topK}} \text{softmax}(\text{Score}_k) \cdot \text{Head}_k(\text{FST}_k(f))$$

where $\text{Score}_k$ incorporates trust, similarity, staleness, AND learned gating. Key differences:

1. **Sparse** (top-K, not all experts) — only K=3 experts contribute per sample
2. **Metadata-informed** — domain knowledge (trust, freshness) is injected directly
3. **Sigmoid per-expert** (not softmax over all) — each expert independently gated
4. **Time-aware** — staleness decay adapts to the temporal dynamics of async training

---

## 6. End-to-End Data Flow

### Forward Pass (Single Training Batch)

```
Input: x = (64, 3, 32, 32), y = (64,)

═══════════════════ LOCAL PATH ═══════════════════
│
│  1. features = body_encoder(x)           → (64, 512)
│  2. local_logits = head(features)        → (64, 10)
│  3. L_local = CrossEntropy(logits, y)    → scalar
│
═══════════════════ MoE PATH ════════════════════
│
│  4. f_det = features.detach()            → (64, 512)  ← GRADIENT CUT
│
│  5. For each expert j in cache:
│     │  aligned_j = FST_j(f_det)          → (64, 512)
│     │  logits_j  = Head_j(aligned_j)     → (64, 10)
│     expert_outputs                       → (64, N_valid, 10)
│
│  6. projected = projector(f_det)         → (64, 256)
│  7. For each expert j:
│     │  sim_j = cos_sim(f_det, repr_j)    → (64,)
│     │  base_j = T_j × max(0,sim_j) × exp(-λΔt_j)  → (64,)
│     │  gate_j = σ(projected · emb_j/√d)  → (64,)
│     │  score_j = gate_j × base_j         → (64,)
│     scores                               → (64, N_valid)
│
│  8. top_scores, top_idx = topk(scores, K=3)  → (64, 3)
│  9. weights = softmax(top_scores)             → (64, 3)
│ 10. selected = gather(expert_outputs, top_idx) → (64, 3, 10)
│ 11. moe_logits = Σ(weights × selected)        → (64, 10)
│ 12. L_MoE = CrossEntropy(moe_logits, y)       → scalar
│
═══════════════════ COMBINED ════════════════════
│
│ 13. L = α · L_local + (1 - α) · L_MoE        → scalar
│ 14. L.backward()
│ 15. clip_grad_norm_(router.params, 1.0)
│ 16. optimizer_head.step()
│     optimizer_body.step()
│     optimizer_router.step()
```

### Gradient Flow Diagram

```
                    L = α · L_local + (1-α) · L_MoE
                          /                    \
                     L_local                  L_MoE
                     /     \                  /    \
               head(f)      f ─── body(x)  router(f.detach())  FSTs
                 │    │      ▲       ▲          │      │
                grad  grad   grad    grad      grad   grad
                 to    to    to      to         to     to
                head  body  body    body      router   FSTs
                             │
                      (from L_local ONLY)
```

**Why `features.detach()` in the MoE path**: Remote expert heads were trained on different class distributions (e.g., Client A trained on classes {0, 1} while Client B has classes {8, 9}). Their predictions for Client B's classes are effectively **random**. If L_MoE gradients flowed back through the body encoder, these random-prediction gradients would **corrupt the body's learned feature space**. The `.detach()` ensures L_MoE only trains the router (feature projector, expert embeddings) and FSTs — components designed to handle cross-client alignment.

---

## 7. Communication Protocol

### Three-Phase Hierarchical Communication

```
Phase 1: BOTTOM-UP (Intra-Cluster)
══════════════════════════════════
Trigger: Every local round (round-based, knowledge-driven)

    client_1 ──expert_pkg──► client_0 (HEAD)
    client_2 ──expert_pkg──► client_0 (HEAD)
    client_0 ──expert_pkg──► client_1, client_2

    All members share with all peers within the cluster.
    Rationale: New knowledge → share immediately to similar peers.


Phase 2: HEAD-TO-HEAD (Cross-Cluster)
═════════════════════════════════════
Trigger: Every 60s (wall-clock time-based, resource-driven)

    HEAD_0 ──own expert + member experts──► HEAD_1, HEAD_2
    HEAD_1 ──own expert + member experts──► HEAD_0, HEAD_2
    HEAD_2 ──own expert + member experts──► HEAD_0, HEAD_1

    Only cluster heads participate. Each head sends:
    1. Its own expert package
    2. All cached intra-cluster member expert packages

    Rationale: Cross-cluster is expensive (head relays ALL member
    experts). Using wall-clock time ensures all clusters share at
    equal intervals regardless of head training speed.


Phase 3: TOP-DOWN RELAY (Head → Members)
════════════════════════════════════════
Trigger: Immediately on receive (event-driven)

    HEAD_0 receives expert from Cluster 1
        └──► relay to client_1, client_2

    When a head receives a cross-cluster expert, it immediately
    broadcasts it to all cluster members. Members never relay
    (prevents infinite loops).
```

### Why Intra-Cluster = Round-Based, Cross-Cluster = Time-Based

| | Intra-Cluster | Cross-Cluster |
|---|---|---|
| **What triggers it** | Client finishes a training round | Wall-clock timer fires |
| **Who controls timing** | Each client individually | One head per cluster |
| **Nature** | Knowledge event ("I learned something new") | Resource event ("time to relay to other clusters") |
| **If round-based** | Natural: share new knowledge immediately | **Problematic**: fast head → floods cross-cluster; slow head → bottlenecks entire cluster |
| **If time-based** | Wasteful: share before learning anything | Natural: normalizes relay frequency across clusters |

The key insight: intra-cluster sharing is **individual** (each client controls its own), but cross-cluster relay is **collective** (one head gates the whole cluster). Individual actions should be knowledge-driven; collective actions should be time-normalized.

### Timing Hierarchy

```
Intra-cluster share:    every round   (~10-20s)    ← knowledge-driven
Cross-cluster relay:    every 60s                   ← resource-driven
Evaluation:             every 120s                  ← system check
Reclustering:           every 150s                  ← structural adaptation
Eviction:               after 300s                  ← disconnect detection
```

Each tier is approximately 2× the previous, creating a natural frequency hierarchy.

---

## 8. True Asynchronous Training

### No Round Barrier

Unlike FedAvg/FedProx/SCAFFOLD (which wait for all clients per round), each client trains independently:

```
                    Wall-Clock Time ───────────────────────────►

Client 0 (fast):  │R1│R2│R3│R4│R5│R6│R7│R8│R9│R10│...│R18│R19│R20│keepalive...│
Client 1:         │R1 │R2 │R3 │R4 │R5 │R6 │R7 │R8 │...│R16│R17│R18│R19│R20│ka..│
Client 2 (slow):  │ R1  │ R2  │ R3  │ R4  │ R5  │ R6  │...│R15│R16│...│R20│ka..│
                  │                                                             │
                  ▼                                                             ▼
              EVAL #1 (120s)                                               EVAL #N

No client waits for any other client. Fast clients complete more rounds
in the same wall-clock time. Expert packages arrive asynchronously.
```

### Keep-Alive Mechanism

When a client finishes all training rounds, it does NOT exit. Instead:

```python
# After training loop completes:
while not all_training_done.wait(timeout=30):
    eval_pause.wait()                           # Pause during evaluation
    client._drain_pending_registrations()        # Keep router fresh
    client.share_expert(force_all_targets=True)  # Re-share to all targets
```

**Why this matters**: Without keep-alive, fast-finishing clients' experts age and get evicted (>300s without refresh). Still-training clients lose valuable experts from their cache, degrading MoE routing quality precisely when they need it most.

**Eviction semantics under keep-alive**:
- Client finishes training → keep-alive → expert stays fresh (**correct**)
- Client crashes/disconnects → no keep-alive → expert ages past 300s → evicted (**correct**)

### Concurrency Model

```
Per Client:
  ┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  Training    │     │  TCP Server  │     │  Handler Threads │
  │  Thread      │     │  Thread      │     │  (per connection)│
  │              │     │              │     │                  │
  │ train_round()│     │ accept_loop()│     │ recv_exact()     │
  │ share_expert │     │   ├─spawn──► │     │ dispatch_message │
  │ keep-alive   │     │   ├─spawn──► │     │ → cache.add()   │
  │              │     │   └─spawn──► │     │ → queue.put()   │
  └──────┬───────┘     └──────────────┘     └────────┬─────────┘
         │                                           │
         │    _pending_registrations (Queue)          │
         │◄──────────────────────────────────────────┘
         │
         │  _drain_pending_registrations()
         │  → router.register_expert()
         │  (serialized with forward_moe reads)

Synchronization Primitives:
  eval_pause:           threading.Event   (pause clients during eval)
  all_training_done:    threading.Event   (signal keep-alive exit)
  stats_queue:          queue.Queue       (metrics → main thread)
  _pending_registrations: queue.Queue     (transport → training thread)
  cache.lock:           threading.RLock   (cache operations)
  peer_lock:            threading.RLock   (socket management)
  send_locks[peer]:     threading.Lock    (per-peer write serialization)
```

---

## 9. Training Pipeline

### Per-Round Training Flow

```
train_round():
  │
  ├─ current_round += 1
  │
  ├─ for epoch in range(local_epochs):     # default: 3
  │    ├─ eval_pause.wait()                # pause if evaluation running
  │    ├─ _drain_pending_registrations()   # register queued experts
  │    ├─ _ensure_fsts_in_optimizer()      # add new FSTs to optimizer
  │    ├─ build frozen expert pool:
  │    │    ├─ frozen local head copy (.eval(), .requires_grad_(False))
  │    │    └─ frozen remote heads from cache (.eval(), .requires_grad_(False))
  │    │
  │    └─ for batch (x, y) in train_loader:
  │         ├─ f = body_encoder(x)                      # (B, 512)
  │         ├─ logits = head(f)                          # (B, 10)
  │         ├─ L_local = CE(logits, y)
  │         ├─ moe_out = router.forward_moe(f.detach(), expert_heads, K)
  │         ├─ L_MoE = CE(moe_out, y)
  │         ├─ alpha = warmup_schedule(current_round)
  │         ├─ L = alpha * L_local + (1 - alpha) * L_MoE
  │         ├─ L.backward()
  │         ├─ clip_grad_norm_(router.params, 1.0)
  │         └─ step all three optimizers
  │
  ├─ validate(val_loader)
  ├─ compute_trust_score(val_acc)
  ├─ compute_representative_features(train_loader, max_batches=5)
  ├─ update cluster_manager
  ├─ re-register local expert in router
  ├─ share_expert()                        # intra-cluster + cross-cluster
  └─ decay learning rates (* 0.98)
```

### Alpha Warmup Schedule (Quadratic)

The hybrid loss weight $\alpha$ starts at 1.0 (pure local) and decays to the target (default 0.5) following a quadratic schedule:

$$\alpha(t) = \alpha_{target} + (1 - \alpha_{target}) \cdot \left(1 - \frac{t}{T_{warmup}}\right)^2$$

```
α
1.0 ─┐
     │ ╲                    Quadratic: MoE gets 2× gradient
     │   ╲                  at round 5 compared to linear
0.8 ─│     ╲
     │       ╲ ╌╌╌╌╌╌╌ Linear (reference)
0.7 ─│         ╲      ╲
     │           ╲       ╲
0.6 ─│─ ─ ─ ─ ─ ─ ╲─ ─ ─ ─╲─ ─ ─ ─ ─ target α
     │               ╲       ╲
     └───┬───┬───┬───┬───┬───┬───┬───►  Round
         1   2   3   4   5   7   10
```

| Round | Linear α | Quadratic α | MoE weight (linear) | MoE weight (quadratic) |
|-------|----------|-------------|----------------------|------------------------|
| 1 | 0.96 | 0.92 | 4% | 8% |
| 3 | 0.88 | 0.80 | 12% | 20% |
| 5 | 0.80 | 0.70 | 20% | 30% |
| 7 | 0.72 | 0.64 | 28% | 36% |
| 10+ | 0.60 | 0.60 | 40% | 40% |

**Why quadratic**: The router and FSTs need gradient signal to learn routing and alignment. Linear warmup gives MoE only 20% weight at the midpoint (round 5/10), starving these components during the critical early learning period. Quadratic gives 30% — significantly more signal when it matters most.

### Three Separate Optimizers

| Optimizer | Parameters | Default LR | Purpose |
|-----------|-----------|------------|---------|
| `optimizer_head` | Head (133,898) | 0.001 | Local classification |
| `optimizer_body` | Body encoder (3,244,864) | 0.001 | Feature extraction |
| `optimizer_router` | Router + FSTs (133,888 + FSTs) | 0.001 | Expert routing & alignment |

All use Adam with weight_decay=1e-4. Learning rates decay by 0.98× per round.

---

## 10. Data Partitioning

**File**: `utils/data_utils.py`

### Dirichlet Distribution Partitioning

For each class $c$:
$$p_c \sim \text{Dir}(\alpha, \alpha, \ldots, \alpha) \quad \text{(N entries)}$$

The proportions $p_c$ determine how samples of class $c$ are divided among N clients.

```
α → 0:    Extreme non-IID (each class goes to one client)
α = 0.1:  Highly skewed (severe heterogeneity)
α = 0.5:  Moderate heterogeneity
α = 1.0:  Mild heterogeneity
α → ∞:    IID (uniform distribution)
```

**Visual example** (α = 0.1, 10 clients, CIFAR-10):

```
         Client 0  Client 1  Client 2  ... Client 9
Class 0: ████████  ░         ░░        ... ░
Class 1: ░         ░░░░░░░░  ░         ... ░
Class 2: ░░        ░         ████████  ... ░
  ...
Class 9: ░         ░         ░         ... ████████

Each client sees only 1-3 classes heavily, others sparsely.
```

### Train/Val Split (Critical Design)

```
1. Partition FULL dataset first (all 50,000 CIFAR-10 training samples)
2. Per client:
     client_indices = partition[i]
     shuffle(client_indices)
     n_val = 10% of |client_indices|
     val_idx = client_indices[:n_val]
     train_idx = client_indices[n_val:]
```

**Why this order matters**: Partitioning first, then splitting ensures validation classes match training classes. The alternative (separate partitioners for train and val) causes class mismatch under label sharding — a client training on {cat, dog} might validate on {truck, airplane}.

---

## 11. Global Evaluation

### Trust-Weighted Confidence Ensemble

With label sharding, each client's individual accuracy on the full 10-class test set is only ~20% (since they only know 2 classes). The global evaluation uses a **trust-weighted confidence ensemble**:

$$\hat{y} = \text{argmax}\left(\frac{\sum_{i=1}^{N} T_i \cdot \text{conf}_i \cdot \text{softmax}(\hat{y}_i)}{\sum_{i=1}^{N} T_i \cdot \text{conf}_i}\right)$$

where $\text{conf}_i = \max_c(\text{softmax}(\hat{y}_i)_c)$ is per-sample confidence.

```
For test sample x (class: "cat"):

  Client 0 (trained on {cat, dog}):
    prediction: [0.85 cat, 0.12 dog, 0.01 ...]   confidence = 0.85
    trust = 0.92    weight = 0.92 × 0.85 = 0.782  ← DOMINATES

  Client 3 (trained on {truck, airplane}):
    prediction: [0.11, 0.09, 0.12, 0.08, ...]     confidence = 0.12
    trust = 0.78    weight = 0.78 × 0.12 = 0.094  ← negligible

  Client 7 (trained on {ship, horse}):
    prediction: [0.08, 0.15, 0.10, ...]            confidence = 0.15
    trust = 0.65    weight = 0.65 × 0.15 = 0.098  ← negligible

  → Ensemble correctly predicts "cat" (dominated by Client 0)
```

This naturally routes each test sample to the most competent client — the one that is both confident (high softmax peak) AND trustworthy (high validation accuracy).

---

## 12. Design Evolution: Trials, Errors & Fixes

This section documents the iterative development process and the reasoning behind key design changes.

### 12.1 Label Sharding + MoE Gradient Corruption

**Problem**: Early versions passed raw features through both the local path and MoE path. Since remote experts were trained on different classes, their predictions for the local client's classes were random. L_MoE gradients flowing back through the body encoder corrupted its feature space.

**Symptom**: Test accuracy plateaued at ~30% despite high per-client val accuracy (~85%).

**Fix**: `features.detach()` before the MoE path. L_MoE now only trains router and FST parameters.

**Lesson**: In heterogeneous FL with MoE, the feature extractor must be protected from gradients originating in foreign experts.

### 12.2 Global Evaluation: Average of Individuals → Ensemble

**Problem**: Early evaluation computed per-client accuracy on the full test set and averaged. With 2/10 classes per client, each scored ~20% → average ~20%.

**Fix**: Trust-weighted confidence ensemble (Section 11). Global accuracy jumped from ~20% to ~60%+.

**Lesson**: Under label sharding, evaluation must leverage the ensemble, not individual clients.

### 12.3 Validation/Training Class Mismatch

**Problem**: Using separate data partitioners for train and val sets caused class mismatch (client trains on {0, 1} but validates on {5, 7}).

**Fix**: Partition first, then split each client's partition into 90% train / 10% val.

### 12.4 FST Parameters Never Updating

**Problem**: FSTs are created lazily (when a new expert arrives), but the optimizer was initialized before any experts were registered. New FST parameters had gradients but were never in any optimizer — they stayed at identity initialization forever.

**Symptom**: FSTs remained identity transforms throughout training, providing no feature alignment.

**Fix**: `_ensure_fsts_in_optimizer()` method that calls `optimizer.add_param_group()` for each new FST. Called at the start of every training epoch.

### 12.5 Synchronous → True Asynchronous

**Problem**: Original design used `for round_num ... as_completed(futures)`, creating a synchronous barrier. All clients had to finish round N before any started round N+1.

**Fix**: Each client runs in its own thread with an independent training loop. No global round counter. Evaluation and reclustering use wall-clock timers instead of round counts.

**Impact**: Clients now naturally train at different speeds. A 17-round spread between fastest and slowest clients was observed in experiments.

### 12.6 Staleness Scoring: λ=0.001 → λ=0.005

**Problem**: With λ=0.001, staleness factors were e^(-0.001×t) ≈ 0.99 for typical expert ages. Zero discriminative power — all experts scored equally regardless of freshness.

**Fix**: Increased to λ=0.005. Now a 30s-old expert scores 0.86 while a 300s-old expert scores 0.22 — meaningful differentiation.

### 12.7 Trust Compression: Smoothed → Direct

**Problem**: Trust = 0.7 × val_acc + 0.3 × min(1, N/10000) gave ~0.75 for all clients. The data_component term compressed everything to near-uniform.

**Fix**: Trust = val_acc directly (clamped to [0.1, 1.0]). Clients with 95% val accuracy get trust 0.95; those with 75% get 0.75 — real differentiation.

### 12.8 Similarity: Compressed → Clamped

**Problem**: `(cosine_sim + 1) / 2` maps [-1, 1] → [0, 1], compressing the useful range. Raw cosines of 0.6–0.9 become 0.8–0.95 (15% spread).

**Fix**: `max(0, cosine_sim)`. Raw cosines stay as 0.6–0.9 (30% spread, 2× more discriminative).

### 12.9 Linear → Quadratic Warmup

**Problem**: Linear warmup from α=1.0 to α=0.5 over 10 rounds gave MoE only 25% gradient weight at round 5. Router and FSTs were gradient-starved during the most critical early learning period.

**Fix**: Quadratic schedule: $(1 - \text{progress})^2$. MoE gets 30% at round 5 (vs 20% linear) — 50% more gradient signal when routing needs to be learned.

### 12.10 Expert Embedding Init: std=0.01 → std=0.1

**Problem**: With std=0.01, the scaled dot product `proj(f) · emb / √256` produced near-zero values → `sigmoid(≈0) ≈ 0.5` for all experts. No initial differentiation.

**Fix**: std=0.1 gives meaningfully different initial embeddings. The gating network can distinguish experts from round 1.

### 12.11 Round-Based → Time-Based Cross-Cluster

**Problem**: `current_round % 5 == 0` makes cross-cluster timing depend on the cluster head's training speed. Fast head → frequent cross-cluster; slow head → entire cluster bottlenecked.

**Fix**: Wall-clock timer (60s default). All clusters share cross-cluster at uniform intervals regardless of head training speed.

### 12.12 Post-Training Expert Eviction → Keep-Alive

**Problem**: When fast clients finished training, they stopped sharing. Their experts aged past the 300s eviction threshold and disappeared from slower clients' caches. Cache shrunk from 9/9 to 3/9.

**Fix**: Keep-alive loop re-shares the final expert every 30s until all clients finish. Three additional fixes in the keep-alive loop:
1. `eval_pause.wait()` — pause sharing during evaluation
2. `_drain_pending_registrations()` — keep router up-to-date
3. `force_all_targets=True` — bypass frozen round modulo logic

### 12.13 EMA Trust Update: Dead Code → Wired

**Problem**: `ExpertMetadata.update_trust()` implemented EMA (0.95 × old + 0.05 × new) but was never called. Trust was directly assigned on each update, causing jumps.

**Fix**: `register_expert()` now calls `update_trust()` instead of direct assignment. Trust evolves smoothly.

### 12.14 Batched MoE Forward: Per-Sample → Batched

**Problem**: Per-sample loop in `forward_moe` computed B×K individual expert forwards. For B=64, K=3, N=10: 192 individual forwards.

**Fix**: Batched approach computes N batched forwards (each processes all B samples at once), then gathers results. 10 batched calls vs 192 individual calls — ~20× faster.

---

## 13. Comparison with Prior Work

### Architectural Comparison

| Framework | Central Server | Synchronous | Full Model Shared | MoE Routing | Non-IID Strategy |
|-----------|:-:|:-:|:-:|:-:|---|
| **FedAvg** (McMahan '17) | Yes | Yes | Yes | No | None (weighted average) |
| **FedProx** (Li '20) | Yes | Yes | Yes | No | Proximal regularization |
| **SCAFFOLD** (Karimireddy '20) | Yes | Yes | Yes + control variates | No | Variance reduction |
| **FedNova** (Wang '20) | Yes | Yes | Yes | No | Normalized averaging |
| **FedMA** (Wang '20) | Yes | Yes | Yes (layer-wise) | No | Neuron matching |
| **FedDF** (Lin '20) | Yes | Yes | Logits | No | Ensemble distillation |
| **D-PSGD** (Lian '17) | No | Yes | Yes (to neighbors) | No | None (gossip averaging) |
| **FedBuff** (Nguyen '22) | Yes | **No** | Yes | No | Buffer averaging + staleness |
| **dFLMoE** (Xie '25) | No | Yes | **Head only** | Yes (cross-attention) | FST + MoE |
| **Ours** | **No** | **No** | **Head only** | **Yes (composite)** | **Trust + Sim + Staleness + Gating + FST + Hierarchy** |

### What Each Framework Lacks

| Framework | Missing vs. Our Approach |
|-----------|--------------------------|
| **FedAvg** | No personalization, no non-IID handling, central server, sync barrier, full model sharing |
| **FedProx** | Single global model with static regularizer, no per-sample routing, central server |
| **SCAFFOLD** | 2× communication overhead (control variates), unstable at 100+ clients, centralized |
| **FedNova** | Addresses computation heterogeneity only (not data heterogeneity), centralized |
| **FedMA** | Expensive neuron matching (O(N³) per layer), architecture-specific, synchronous |
| **FedDF** | Requires auxiliary unlabeled dataset, centralized distillation |
| **D-PSGD** | Full model gossip, fixed topology, no trust/staleness, sensitive to non-IID |
| **FedBuff** | Centralized buffer, simple polynomial staleness (vs our multiplicative composite), no MoE |
| **dFLMoE** | Synchronous rounds, no trust scoring, no staleness decay, no hierarchical clustering, no per-sample gating |

### Detailed Comparison with dFLMoE (Xie et al., CVPR 2025)

dFLMoE is the most closely related prior work as it also uses decentralized head sharing with FSTs:

| Aspect | dFLMoE (Xie et al.) | Our Framework |
|--------|---------------------|---------------|
| **Synchrony** | Synchronous rounds | True asynchronous |
| **Routing mechanism** | Cross-attention: $y = \text{Attn}(W \cdot I, K, V)$ | Composite: $\sigma(\text{attn}) \times T \times S \times e^{-\lambda t}$ |
| **Trust scoring** | Not present | EMA-smoothed per-expert trust from val accuracy |
| **Staleness handling** | Not addressed | Exponential decay $e^{-\lambda \Delta t}$ |
| **Temporal dynamics** | Static (round-based) | Dynamic (experts age, arrive, get evicted) |
| **Communication topology** | Flat (all-to-all) | Hierarchical (K-Means clusters + head relay) |
| **Expert management** | Static pool per round | Dynamic cache with staleness eviction (300s) |
| **Gating** | Dense cross-attention (all experts) | Sparse top-K with sigmoid per-expert |
| **Feature alignment** | FST (shared) | FST (same concept, identity-init) |
| **Scalability** | O(N²) communication | O(N) with hierarchical clustering |

**Key differentiator**: dFLMoE uses a standard cross-attention mechanism where the model must learn ALL routing behavior from data. Our approach injects **domain knowledge** directly into the scoring formula — trust, similarity, and staleness are computed from metadata, not learned. The learned gating only needs to capture residual patterns that metadata cannot express. This makes routing more sample-efficient and robust.

### Communication Cost Comparison

| Framework | Data Transmitted per Round | With SimpleCNNBody (3.4M params) |
|-----------|---------------------------|----------------------------------|
| FedAvg | Full model × N | 3.4M × 10 = **34M params** |
| SCAFFOLD | Full model × N + control variates × N | **68M params** |
| FedMA | Full model × N (layer-wise) | **34M params** |
| D-PSGD | Full model × neighbors | ~3.4M × 3 = **10.2M params** |
| **Ours** | Head only × cluster peers | ~134K × 3 = **0.4M params** (96% reduction vs FedAvg) |

---

## 14. Experiment Results

> **Note**: The results below are from experiments run during development. The framework has undergone significant architectural changes since these experiments (true async, time-based cross-cluster, keep-alive, etc.). Final results with the current codebase will be updated.

### CIFAR-10 Results (Historical, Synchronous Setup)

#### Varying Dirichlet α (20 rounds, WD=0.0001, LR_decay=0.98, Dropout=0.3)

| Dirichlet α | Non-IID Level | Final Test Acc | Best Test Acc | Training Time |
|-------------|---------------|----------------|---------------|---------------|
| 0.1 | Extreme | 63.23% | 63.72% | 103.7 min |
| 0.2 | High | 71.22% | 71.13% | 106.3 min |
| 0.3 | Moderate-High | 73.31% | 73.48% | 102.8 min |
| 0.5 | Moderate | 77.88% | **78.45%** | 130.0 min |

#### Varying Training Rounds (α=0.3, WD=0.0001, LR_decay=0.98, Dropout=0.3)

| Rounds | Final Test Acc | Best Test Acc | Training Time |
|--------|----------------|---------------|---------------|
| 20 | 73.31% | 73.48% | 102.8 min |
| 30 | 75.65% | **76.50%** | 155.3 min |
| 40 | 75.30% | 76.04% | 205.8 min |

#### Latest Asynchronous Experiment (α=0.1, 20 rounds)

| Metric | Value |
|--------|-------|
| Final test accuracy | 63.79% |
| Best test accuracy | **64.46%** |
| Training time | 140.1 min |
| Round spread (fastest − slowest) | 17 rounds |
| Learning curve | Continuous upward (35% → 64.5%, no plateau) |

### MNIST Result (α=0.2, 20 rounds)

| Metric | Value |
|--------|-------|
| Best test accuracy | **97.76%** |

### Key Observations

1. **Non-IID severity is the dominant factor**: α=0.5 achieves 78.45% while α=0.1 achieves 64.46% — a direct consequence of data heterogeneity.

2. **Async matches sync quality**: The latest async result (64.46% at α=0.1) is comparable to the sync result (63.72%), confirming async training does not degrade accuracy.

3. **No plateau at 20 rounds**: The learning curve shows continuous improvement at the final evaluation, suggesting higher accuracy with more rounds.

4. **Diminishing returns beyond 30 rounds**: At α=0.3, going from 30→40 rounds yields no improvement, likely due to local overfitting.

---

## 15. Setup & Usage

### Requirements

```
Python >= 3.8
PyTorch >= 1.10
scikit-learn
numpy
```

### Running an Experiment

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

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.5 | Local loss weight (1-α goes to MoE) |
| `--non_iid_alpha` | 0.5 | Dirichlet concentration (lower = more heterogeneous) |
| `--staleness_lambda` | 0.005 | Staleness decay rate λ |
| `--max_expert_age` | 300 | Expert eviction threshold (seconds) |
| `--cross_cluster_interval` | 60 | Cross-cluster exchange interval (seconds) |
| `--eval_interval` | 300 | Global evaluation interval (seconds) |
| `--recluster_interval` | 150 | K-Means reclustering interval (seconds) |
| `--top_k_experts` | 3 | Number of experts selected per sample |
| `--warmup_rounds` | 10 | Rounds for alpha warmup |

---

## 16. References

1. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
2. Li, T., et al. "Federated Optimization in Heterogeneous Networks." MLSys 2020.
3. Karimireddy, S.P., et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020.
4. Wang, H., et al. "Federated Learning with Matched Averaging." ICLR 2020.
5. Wang, J., et al. "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." NeurIPS 2020.
6. Lin, T., et al. "Ensemble Distillation for Robust Model Fusion in Federated Learning." NeurIPS 2020.
7. Lian, X., et al. "Can Decentralized Algorithms Outperform Centralized Algorithms?" NeurIPS 2017.
8. Nguyen, J., et al. "Federated Learning with Buffered Asynchronous Aggregation." AISTATS 2022.
9. Xie, Z., et al. "dFLMoE: Decentralized Federated Learning via Mixture of Experts." CVPR 2025.
10. Fallah, A., et al. "Personalized Federated Learning with Theoretical Guarantees." NeurIPS 2020.

---

## Project Structure

```
Adaptive-Asynchronous-Hierarchical-dFLMoE/
├── main.py                    # Orchestration: data loading, async training, evaluation
├── client_node.py             # Client: training loop, hybrid loss, expert sharing
├── models/
│   ├── body_encoder.py        # SimpleCNNBody (private feature extractor)
│   ├── head.py                # Expert Head (shared lightweight classifier)
│   ├── fst.py                 # Feature Space Transform (identity-init alignment)
│   └── router.py              # Router (composite scoring + MoE aggregation)
├── infra/
│   ├── peer_cache.py          # Thread-safe expert cache with staleness eviction
│   ├── transport.py           # TCP P2P transport (real sockets, not simulation)
│   └── cluster.py             # K-Means hierarchical clustering
├── utils/
│   └── data_utils.py          # Dirichlet/label_sharding/IID data partitioning
└── results/                   # Experiment results
```
