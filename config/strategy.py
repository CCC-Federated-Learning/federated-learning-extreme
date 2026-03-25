# ============================================================================
# STRATEGY-SPECIFIC CONFIGURATION MODULE
# ============================================================================
# This module centralizes all strategy hyperparameters.
# Each strategy section includes:
#   - Parameter definitions
#   - Recommended values for common scenarios
#   - Functional explanations
#   - Dependency notes (e.g., which strategies use which parameters)
#
# IMPORTANT: You can set some values to None to fallback to global config defaults
#            (see config/__init__.py for fallback logic).
#
# For detailed strategy analysis, see: ../STRATEGY_AUDIT_LOG.md

# STRATEGIES_IN_ORDER = [
#     "fedavg",
#     "fedavgm",
#     "fedadagrad",
#     "fedadam",
#     "fedprox",
#     "fedyogi",
#     "bulyan",
#     "krum",
#     "multikrum",
#     "fedmedian",
#     "fedtrimmedavg",
#     "differentialprivacyclientsideadaptiveclipping",
#     "differentialprivacyclientsidefixedclipping",
#     "differentialprivacyserversideadaptiveclipping",
#     "differentialprivacyserversidefixedclipping",
#     "fedxgbbagging",
#     "fedxgbcyclic",
#     "qfedavg",
# ]

# ============================================================================
# 1️⃣ FEDAVG - Federated Averaging (Standard FL Baseline)
# ============================================================================
# Computes simple weighted average of client updates on the server.
# No special hyperparameters - depends only on global settings.
#
# Use when: IID data, homogeneous clients, baseline comparison
# 
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE (from config/__init__.py)
# Python class: flwr.serverapp.strategy.FedAvg
# Reference: "Communication-Efficient Learning of Deep Networks from Decentralized Data"


# ============================================================================
# 2️⃣ FEDAVGM - Federated Averaging with Momentum
# ============================================================================
# Adds server-side momentum to stabilize convergence on non-convex problems.
# Maintains a velocity vector updated with server gradients.
#
# Use when: Non-IID data, general federated learning, faster convergence desired
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE + SERVER_LEARNING_RATE, SERVER_MOMENTUM
# Python class: flwr.serverapp.strategy.FedAvgM
# Reference: "Adaptive Federated Optimization" (Reddi et al., 2020)

SERVER_LEARNING_RATE = 1.0
# ├─ Purpose: Server-side optimizer learning rate (independent from client LR)
# ├─ Type: float
# ├─ Range: [0.1, 5.0] typical; [0.5, 2.0] recommended
# ├─ Effect: Controls magnitude of server parameter updates
# │   - Too high (>2.0) → divergence, loss spikes
# │   - Too low (<0.1) → very slow convergence
# ├─ Relationship to client LR: INDEPENDENT
# │   Client LR ∈ [0.001, 0.1]; Server LR ∈ [0.1, 5.0]
# ├─ Scenario 1 (Good convergence): Set to 1.0-1.5
# ├─ Scenario 2 (Unstable loss): Reduce to 0.5-0.8
# ├─ Scenario 3 (Very slow): Increase to 2.0-3.0
# └─ Note: Monitor loss curve; should see steady decrease

SERVER_MOMENTUM = 0.9
# ├─ Purpose: Momentum coefficient for server-side updates
# ├─ Type: float
# ├─ Range: [0.0, 1.0]
# ├─ Effect on convergence:
# │   v_t = β * v_{t-1} + ∇L_t  (β = SERVER_MOMENTUM)
# │   - β=0.0 → pure gradient descent (no acceleration)
# │   - β=0.9 → strong momentum (typical)
# │   - β≥1.0 → undefined (will cause divergence)
# ├─ Recommended value: 0.9 (standard in deep learning)
# ├─ Common range: [0.8, 0.99]
# └─ Warning: Rarely needs adjustment; keep 0.9 unless debugging


# ============================================================================
# 3️⃣ FEDPROX - Federated Proximal
# ============================================================================
# Adds a proximal regularization term to prevent clients from diverging
# too far from the server model. Useful for non-IID data and heterogeneous clients.
#
# Use when: Highly non-IID data, heterogeneous clients, data drift scenarios
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE + PROXIMAL_MU
# Client-side modification: YES (computes proximal term in loss)
# Python class: flwr.serverapp.strategy.FedProx
# Reference: "Federated Optimization in Heterogeneous Networks" (Li et al., 2018)

PROXIMAL_MU = 0.1
# ├─ Purpose: Proximal regularization strength
# ├─ Type: float
# ├─ Range: [0.0, 1.0] typical
# ├─ Effect: Constrains client parameters to stay close to server
# │   Loss_client = Loss_original + (μ/2) * ||θ_client - θ_server||²
# │                                           ↑ proximal constraint term
# ├─ Tuning by data heterogeneity (Dirichlet α):
# │   - IID (α→∞)           → PROXIMAL_MU = 0.01-0.05
# │   - Moderate (α=0.5)    → PROXIMAL_MU = 0.1-0.3  (default)
# │   - Extreme (α<0.1)     → PROXIMAL_MU = 0.5-1.0
# ├─ Effect of different values:
# │   - 0.0: Proximal term disabled, reverts to FedAvg
# │   - 0.01-0.1: Gentle constraint, good for IID data
# │   - 0.1-0.3: Moderate constraint, balanced (most common)
# │   - 0.5-1.0: Strong constraint, for highly non-IID
# │   - >1.0: Over-constrained, slow convergence
# ├─ Client-side implication:
# │   - Must save server parameters before training
# │   - Add proximal term to loss: loss += (μ/2) * sum((p - p_server)²)
# │   - Increases per-client computation (minor overhead)
# ├─ Convergence behavior:
# │   FedProx typically converges slightly slower than FedAvg on IID,
# │   but much better on non-IID
# └─ Note: Value affects both Server (FedProx strategy) and Client (loss computation)


# ============================================================================
# 4️⃣ FEDADAGRAD - Adaptive Federated Averaging (Server-side AdaGrad)
# ============================================================================
# Server-side adaptive optimization using Adagrad algorithm.
# Each parameter gets its own adaptive learning rate based on historical gradients.
#
# Key Insight: Client trains normally → Server treats updates as "pseudo-gradients"
#              → Server's Adagrad optimizer adapts per-parameter LR
#
# Use when: Non-IID data with varying client gradient scales, fast convergence needed
# Robustness: Better than FedAvg for heterogeneous data; requires careful tuning
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE + SERVER_ETA, SERVER_ETA_L, SERVER_TAU
# Client modification: NO (standard PyTorch training only)
# Python class: flwr.serverapp.strategy.FedAdagrad
# Reference: "Adaptive Federated Optimization" (Reddi et al., 2020, https://arxiv.org/abs/2003.00295v5)
#
# ⚠️ CRITICAL: The parameter SERVER_ETA_L is used on SERVER SIDE, NOT client side.
#    It's part of the mathematical formulation for "undoing" client updates.
#    CLIENT SIDE: Always use your configured LR (e.g., 0.001)
#    SERVER SIDE: Flower auto-handles SERVER_ETA_L in the aggregation algorithm


# ============================================================================
# 5️⃣ FEDADAM - Federated Adam
# ============================================================================
# Server-side adaptive optimization using Adam algorithm.
# Combines:
#   - First moment (exponential moving average of gradients)
#   - Second moment (exponential moving average of squared gradients)
#
# Use when: Complex loss landscapes, faster convergence on non-IID data
# Robustness: Similar to FedAdagrad; may be more stable with proper hyperparameter tuning
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE + SERVER_ETA, SERVER_ETA_L, 
#               SERVER_TAU, BETA_1, BETA_2
# Client modification: NO (standard PyTorch training only)
# Python class: flwr.serverapp.strategy.FedAdam
# Reference: "Adaptive Federated Optimization" (Reddi et al., 2020)
#
# ⚠️ WARNING: FedAdam > FedAdagrad in most scenarios
#    If FedAdagrad converges slowly, upgrade to FedAdam before tuning hyperparams


# ============================================================================
# 6️⃣ FEDYOGI - Federated Yogi (Improved Adaptive Optimizer)
# ============================================================================
# Server-side adaptive optimization using Yogi algorithm.
# Yogi is an IMPROVED version of Adam, better at handling:
#   - Extreme gradient values (outliers)
#   - Highly non-IID data (heterogeneous clients)
#
# Key difference from Adam:
#   - Uses element-wise MINIMUM for second moment (instead of EXPONENTIAL moving avg)
#   - More stable in adversarial federated settings
#   - Better resistance to gradient explosions/vanishing
#
# Use when: Extremely non-IID data, training becomes unstable
# Robustness: HIGHEST among adaptive optimizers; specifically designed for federated settings
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE + SERVER_ETA, SERVER_ETA_L,
#               SERVER_TAU, BETA_1, BETA_2
# Client modification: NO (standard PyTorch training only)
# Python class: flwr.serverapp.strategy.FedYogi
# Reference: "Adaptive Methods for Nonconvex Optimization" (Reddi et al., 2018)
#
# 🌟 RECOMMENDATION: If FedAdam fails or diverges → Try FedYogi
#    FedYogi handles edge cases better but slightly slower convergence on well-behaved data


# Adaptive optimizers (FedAdaGrad, FedAdam, FedYogi) shared parameters
SERVER_ETA = 0.1
# ├─ Purpose: Server-side learning rate for adaptive optimizers
# ├─ Type: float
# ├─ Range: [0.001, 1.0] typical
# ├─ Physics: Contols step size in server optimization
# │   Example: New_weights = Old_weights - SERVER_ETA * Adaptive_gradient
# ├─ How to tune:
# │   1. Start with 0.1 (good default)
# │   2. If loss oscillates → decrease to 0.01-0.05
# │   3. If convergence very slow → increase to 0.5-1.0
# ├─ Typical values by optimizer:
# │   FedAdagrad  → 0.1 (balanced)
# │   FedAdam     → 0.01 (more conservative)
# │   FedYogi     → 0.01 (more conservative)
# ├─ Common errors:
# │   ❌ Setting too high (>1.0) → Loss becomes NaN/Inf
# │   ❌ Setting too low (<0.001) → Convergence fails
# │   ✅ Start with 0.1, monitor loss curve, adjust ±5x
# └─ Note: DIFFERENT from client-side LR (0.001-0.1); this is SERVER-SIDE only

SERVER_ETA_L = 0.1
# ├─ Purpose: Client learning rate equivalent used by SERVER aggregation
# ├─ Type: float
# ├─ Range: [0.001, 1.0] typical
# ├─ ⚠️ CRITICAL CLARIFICATION:
# │   This parameter is used ONLY on the server side in the FedOpt aggregation.
# │   It does NOT appear in client training code.
# │   Its mathematical role: "undoes" the effect of client updates in the gradient formula.
# │   It is NOT the client's actual learning rate (which is configured separately).
# ├─ Mathematical context (in FedAdam/Adagrad/Yogi):
# │   pseudo-gradient = (global_params - client_params) / SERVER_ETA_L
# │                     ↑ This rescaling is where SERVER_ETA_L is used
# ├─ How to set:
# │   Option 1: Set to your typical client learning rate (0.001) for consistency
# │   Option 2: Set to a fixed "reference" LR (e.g., 0.1) if clients have varying LRs
# │   Recommendation: Use Option 1 for most scenarios
# ├─ Typical values:
# │   If client LR is 0.001  → Set SERVER_ETA_L = 0.001
# │   If client LR is 0.01   → Set SERVER_ETA_L = 0.01
# │   If variable across clients → Use average or typical client LR
# ├─ Common pitfall:
# │   ❌ Setting SERVER_ETA_L ≠ client LR → Pseudo-gradients misaligned → Poor convergence
# │   ✅ Keep SERVER_ETA_L close to your client learning rate
# └─ Note: Unlike SERVER_ETA, this parameter is less frequently tuned

SERVER_TAU = 0.001
# ├─ Purpose: Regularization constant to prevent division by zero
# ├─ Type: float (small positive)
# ├─ Range: [1e-6, 1e-2] typical
# ├─ Physics: Added to adaptive learning rate denominator
# │   Example (Adam): adaptive_lr = SERVER_ETA / (sqrt(second_moment) + SERVER_TAU)
# │                   Without tau, division by zero if second_moment=0
# ├─ Effect of different values:
# │   Very small (1e-6)  → Adaptive optimizer very sensitive, may overflow
# │   Small (1e-4)       → Standard (balanced)
# │   Moderate (1e-3)    → More regularization (safer)
# │   Large (1e-2)       → Heavy regularization (less adaptation)
# ├─ Typical scenarios:
# │   FedAdagrad → 1e-3 or 1e-4
# │   FedAdam    → 1e-3 or 1e-2
# │   FedYogi    → 1e-3 (default recommendation)
# ├─ When to adjust:
# │   - If gradients become NaN/Inf → increase tau to 1e-2
# │   - If convergence stalls → decrease tau to 1e-5
# │   - Usually don't need adjustment initially
# └─ Rule of thumb: Keep between 1e-5 and 1e-2; 1e-3 is safe default

BETA_1 = 0.9
# ├─ Purpose: Exponential decay rate for first moment (momentum)
# ├─ Type: float
# ├─ Range: [0.8, 0.999]
# ├─ Physics: v_t = BETA_1 * v_{t-1} + (1 - BETA_1) * gradient_t
# │   (exponential moving average of gradients)
# ├─ Effect:
# │   - BETA_1 → 0.0: Pure gradient descent (no momentum)
# │   - BETA_1 = 0.9: Standard (good balance)
# │   - BETA_1 → 1.0: Strong momentum (risky, can oscillate)
# ├─ Used by: FedAdam, FedYogi (NOT FedAdagrad)
# ├─ Typical:
# │   Standard setting: 0.9
# │   Conservative: 0.95
# │   Aggressive: 0.8
# └─ ⚠️: Very rarely needs adjustment; leave at 0.9 unless debugging

BETA_2 = 0.99
# ├─ Purpose: Exponential decay rate for second moment
# ├─ Type: float
# ├─ Range: [0.99, 0.999]
# ├─ Physics: v2_t = BETA_2 * v2_{t-1} + (1 - BETA_2) * gradient_t²
# │   (exponential moving average of squared gradients)
# ├─ Effect:
# │   - BETA_2 → 0.9: Rapid second-moment updates (unstable)
# │   - BETA_2 = 0.99: Standard
# │   - BETA_2 = 0.999: Very stable but slower adaptation
# ├─ Used by: FedAdam, FedYogi (NOT FedAdagrad)
# ├─ Typical values:
# │   FedAdam: 0.99 (balanced)
# │   FedYogi: 0.99 (balanced, rarely changed)
# ├─ How to adjust (rarely needed):
# │   - If oscillations occur → increase to 0.999
# │   - If stagnates early → decrease to 0.95
# └─ Rule: Keep high (>0.9) to stabilize second moment estimation


# ============================================================================
# 7️⃣ FEDMEDIAN - Federated Median (Statistical Robust Aggregation)
# ============================================================================
# Robust aggregation by taking coordinate-wise median across client updates.
#
# Key idea:
#   Instead of averaging all updates (FedAvg), each parameter dimension uses median,
#   which naturally suppresses extreme outliers.
#
# Use when:
#   - You expect occasional anomalous client updates
#   - You want simple robustness without Byzantine distance calculations
#
# Trade-off:
#   - More robust to outliers than FedAvg
#   - Can converge slower because coordinate-wise median may break parameter correlations
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE (global)
# Client modification: NO (standard PyTorch training)
# Python class: flwr.serverapp.strategy.FedMedian
# Reference: "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"


# ============================================================================
# 8️⃣ FEDTRIMMEDAVG - Federated Trimmed Mean
# ============================================================================
# Robust aggregation that removes extreme updates before averaging.
#
# Key idea:
#   For each parameter dimension, sort client values and trim both tails,
#   then average the remaining middle values.
#
# Compared to FedMedian:
#   - FedMedian keeps only middle value (max robustness, slower)
#   - FedTrimmedAvg keeps middle range (better utility/accuracy trade-off)
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE + FEDTRIMMEDAVG_BETA
# Client modification: NO (standard PyTorch training)
# Python class: flwr.serverapp.strategy.FedTrimmedAvg
# Reference: https://arxiv.org/abs/1803.01498

FEDTRIMMEDAVG_BETA = 0.2
# ├─ Purpose: Fraction trimmed from each tail (low and high) per dimension
# ├─ Type: float
# ├─ Range: [0.0, 0.5)  (strictly less than 0.5)
# ├─ Interpretation:
# │   - beta=0.0  → no trimming (close to FedAvg behavior)
# │   - beta=0.1  → trim lowest 10% + highest 10%
# │   - beta=0.2  → trim lowest 20% + highest 20% (default robust setting)
# ├─ Recommended values:
# │   - IID / mild non-IID: 0.05 ~ 0.1
# │   - Moderate non-IID:   0.1  ~ 0.2
# │   - Extreme non-IID:    start low (0.05), avoid aggressive trimming
# ├─ Practical risk:
# │   On label-skewed non-IID data, genuine client updates may look like outliers.
# │   If beta is too large, useful signal can be over-trimmed and learning may stall.
# └─ Tuning hint:
#    If loss plateaus early, reduce beta first (e.g., 0.2 → 0.1 → 0.05)


# ============================================================================
# 9️⃣ ROBUST AGGREGATION - Byzantine-Resilient Strategies
# ============================================================================
# For defense against malicious clients. Require assumption about max malicious clients.

BULYAN_NUM_MALICIOUS_NODES = 0
# ├─ Purpose: Expected number of malicious clients for Bulyan defense
# ├─ Type: int
# ├─ Range: [0, NUM_PARTITIONS-2]
# ├─ Note: Set to 0 if no Byzantine clients expected
# └─ Used by: FedBulyan strategy

KRUM_NUM_MALICIOUS_NODES = 0
# ├─ Purpose: Expected number of malicious clients for Krum defense
# ├─ Type: int
# ├─ Range: [0, NUM_PARTITIONS-2]
# ├─ Warning: Krum is computationally expensive; use only if necessary
# └─ Used by: FedKrum strategy

MULTIKRUM_NUM_MALICIOUS_NODES = 0
# ├─ Purpose: Expected number of malicious clients for MultiKrum defense
# ├─ Type: int
# └─ Used by: FedMultiKrum strategy

MULTIKRUM_NUM_NODES_TO_SELECT = 1
# ├─ Purpose: Number of subsets to aggregate (MultiKrum parameter)
# ├─ Type: int
# ├─ Range: [1, NUM_PARTITIONS]
# └─ Note: Higher values increase computational cost but robustness


# ============================================================================
# 1️⃣0️⃣ DIFFERENTIAL PRIVACY - Privacy-Preserving Federated Learning
# ============================================================================
# Adds noise to gradients to provide formal privacy guarantees.
# Four variants: Client-side (adaptive/fixed) and Server-side (adaptive/fixed)
#
# Wrapper mapping in this project:
#   - DifferentialPrivacyClientSideAdaptiveClipping  -> strategies/dp_client_adaptive.py
#   - DifferentialPrivacyClientSideFixedClipping     -> strategies/dp_client_fixed.py
#   - DifferentialPrivacyServerSideAdaptiveClipping  -> strategies/dp_server_adaptive.py
#   - DifferentialPrivacyServerSideFixedClipping     -> strategies/dp_server_fixed.py
#
# Clipping responsibility:
#   - Client-side variants: clipping happens in app/client.py (manual equivalent of DP mods)
#   - Server-side variants: clipping/noise handled by server wrapper
#
# IMPORTANT:
#   DP_NUM_SAMPLED_CLIENTS should match the expected sampled train clients per round.
#   In this project, expected round sample count is approximately:
#       n = max(2, int(NUM_PARTITIONS * FRACTION_TRAIN))
#   Mismatch can cause incorrect privacy accounting.

DP_NOISE_MULTIPLIER = 0.5
# ├─ Purpose: Noise scale relative to sensitivity (controls privacy level)
# ├─ Type: float
# ├─ Range: [0.01, 10.0]
# ├─ Effect on privacy: Lower value → stronger privacy, but lower accuracy
# │   - 0.01-0.1: Very strong privacy (δ ≈ 10^{-5})
# │   - 0.1-1.0: Strong privacy (δ ≈ 10^{-3})
# │   - 1.0-5.0: Moderate privacy
# │   - >5.0: Weak privacy (almost no effect)
# ├─ Convergence: Higher noise → more rounds needed for same accuracy
# └─ Note: Interact with DP_CLIPPING_NORM; lower clipping → need higher noise

DP_CLIPPING_NORM = 1.0
# ├─ Purpose: Norm bound for gradient clipping (limits sensitivity)
# ├─ Type: float
# ├─ Effect: Prevents unbounded gradients; activates privacy noise
# │   - Lower value→stronger clipping, need higher noise, better privacy
# │   - Higher value→weaker clipping, lower noise, but less privacy
# │   - Typical: 1.0-100.0 depending on problem scale
# ├─ Tuning:
# │   1. Start with 1.0
# │   2. If accuracy drops drastically, increase to 10.0 or 100.0
# │   3. Increase DP_NOISE_MULTIPLIER if privacy too weak
# └─ Used by: All DP strategies (both client-side and server-side)

DP_NUM_SAMPLED_CLIENTS = None  # Fallback to NUM_PARTITIONS if None
# ├─ Purpose: Number of clients per round for DP (subset sampling)
# ├─ Type: int or None
# ├─ If None: Falls back to NUM_PARTITIONS (uses all clients)
# ├─ Effect: Affects privacy amplification by subsampling
# │   - Smaller value→better privacy via amplification, but more rounds
# │   - None→strongest privacy guarantee, but no subsampling adjustment
# └─ Advanced: Leave as None unless privacy calculations are customized

DP_INITIAL_CLIPPING_NORM = 0.1
# ├─ Purpose: Initial clipping norm for adaptive strategies (before adaptation)
# ├─ Type: float
# ├─ Used by: DifferentialPrivacyClientSideAdaptiveClipping,
# │           DifferentialPrivacyServerSideAdaptiveClipping
# └─ Typical: 0.1-1.0

DP_TARGET_CLIPPED_QUANTILE = 0.5
# ├─ Purpose: Target quantile for adaptive clipping norm updates
# ├─ Type: float (in [0, 1])
# ├─ Effect: Adaptive strategies adjust clipping to hit this quantile
# │   - 0.5 → median norm becomes clipping threshold
# │   - Higher (0.7-0.9) → more clipping, stronger privacy
# │   - Lower (0.1-0.3) → less clipping, better utility
# ├─ Interpretation: After clipping, this fraction of updates are fully clipped
# └─ Used by: *AdaptiveClipping strategies

DP_CLIP_NORM_LR = 0.2
# ├─ Purpose: Learning rate for adaptive clipping norm update
# ├─ Type: float
# ├─ Effect: How quickly adaptation adjusts the clipping norm
# │   - Low (0.01-0.1) → slow adaptation, stable
# │   - High (0.2-0.5) → fast adaptation, responsive
# └─ Used by: *AdaptiveClipping strategies

DP_CLIPPED_COUNT_STDDEV = None  # None = no noisy count, or value for Gaussian noise
# ├─ Purpose: Std dev for noisy count (privacy on client count itself)
# ├─ Type: float or None
# ├─ If None: Do not add noise to clipped count, only to gradients
# └─ Advanced: Leave as None for typical use cases


# ============================================================================
# 1️⃣1️⃣ QFEDAVG - Quantile-weighted Federated Averaging
# ============================================================================
# Fairness-oriented federated optimization.
#
# Core idea:
#   QFedAvg increases the contribution of higher-loss clients during aggregation,
#   reducing performance disparity across heterogeneous clients.
#
# Practical meaning:
#   - q = 0   -> equivalent to FedAvg
#   - larger q -> stronger fairness pressure (focus on worse-off clients)
#
# Dependencies: FRACTION_TRAIN, FRACTION_EVALUATE + QFEDAVG_CLIENT_LEARNING_RATE, QFEDAVG_Q
# Client requirement: train metrics must include key "train_loss" (already provided by app/client.py)
# Python class: flwr.serverapp.strategy.QFedAvg
# Reference: https://openreview.net/pdf?id=ByexElSYDr

QFEDAVG_CLIENT_LEARNING_RATE = None  # Fallback to LR if None
# ├─ Purpose: Client learning rate used by QFedAvg to estimate Lipschitz constant
# ├─ Type: float or None
# ├─ If None: Falls back to global LR
# ├─ QFedAvg internal relation: L ≈ 1 / client_learning_rate
# ├─ Typical: same as actual client LR (recommended)
# ├─ If mismatched from true client LR: fairness scaling can be distorted
# └─ Recommendation: keep equal to effective client training LR

QFEDAVG_Q = 0.01
# ├─ Purpose: Quantile parameter for weighted aggregation
# ├─ Type: float
# ├─ Range: [0.0, +inf)   (practically 0.0~1.0 is common)
# ├─ Effect: Emphasizes poorly performing clients
# │   - q=0.0 → FedAvg behavior (no fairness correction)
# │   - q=0.1 → mild fairness correction (default)
# │   - q=0.5 → strong fairness correction
# │   - q>=1.0 → very aggressive; may hurt global average accuracy
# ├─ Tuning advice:
# │   1) Start with 0.1
# │   2) If low-performing clients still lag, increase to 0.2~0.5
# │   3) If overall accuracy drops too much, reduce q
# └─ Note: Larger q often improves fairness but can slow or destabilize convergence


# ============================================================================
# 1️⃣2️⃣ XGBOOST - Tree-Based Federated Learning
# ============================================================================
# FedXGBBagging and FedXGBCyclic: Two approaches to federated XGBoost.

XGB_NUM_LOCAL_ROUND = 1
# ├─ Purpose: Number of boosting rounds per client per FL round
# ├─ Type: int
# ├─ Range: [1, 20]
# ├─ Effect: More rounds→more local model improvement, slower communication
# └─ Typical: 1-5

XGB_MAX_DEPTH = 6
# ├─ Purpose: Maximum tree depth in XGBoost
# ├─ Type: int
# ├─ Range: [2, 10]
# ├─ Effect: Deeper trees → higher capacity but overfitting risk
# └─ Typical: 6-8 for MNIST, 3-5 for small datasets

XGB_ETA = 0.3
# ├─ Purpose: Learning rate (shrinkage factor) for XGBoost
# ├─ Type: float
# ├─ Range: [0.01, 1.0]
# ├─ Effect: Lower value→slower but more stable learning
# │   - 0.3 is standard; often 0.1-0.3 for robust training
# │   - 0.01 → very slow convergence but stable
# │   - 0.5 → fast but risky
# └─ Typical: 0.3 (default)

XGB_SUBSAMPLE = 0.8
# ├─ Purpose: Fraction of samples to use for each tree
# ├─ Type: float
# ├─ Range: [0.5, 1.0]
# ├─ Effect: <1.0 introduces randomness, reduces overfitting
# │   - 1.0 → use all data (deterministic)
# │   - 0.8 → sample 80% of data (random forest-like)
# └─ Typical: 1.0 for federated setting (no subsampling)

XGB_COLSAMPLE_BYTREE = 0.8
# ├─ Purpose: Fraction of features to use for each tree
# ├─ Type: float
# ├─ Range: [0.3, 1.0]
# ├─ Effect: <1.0 adds feature subsampling, improves generalization
# │   - 1.0 → use all features
# │   - 0.8 → random 80% of features per tree
# └─ Typical: 0.8 (prevents overfitting)

XGB_MIN_CHILD_WEIGHT = 1.0
# ├─ Purpose: Minimum sum of weights needed in a child node
# ├─ Type: float
# ├─ Range: [0.0, 10.0]
# ├─ Effect: Prevents overfitting to small node populations
# │   - 1.0 → no constraint (or minimal)
# │   - >5.0 → strong constraint, may underfit
# └─ Typical: 1.0-5.0

XGB_REG_LAMBDA = 1.0
# ├─ Purpose: L2 regularization weight
# ├─ Type: float
# ├─ Range: [0.0, 10.0]
# ├─ Effect: Regularizes tree weights to prevent overfitting
# │   - 0.0 → no regularization
# │   - 1.0 → moderate regularization (default)
# │   - 5.0-10.0 → strong regularization
# └─ Typical: 1.0-2.0


# ============================================================================
# END OF STRATEGY CONFIGURATION
# ============================================================================
# For detailed guidance on each strategy, see: ../STRATEGY_AUDIT_LOG.md

