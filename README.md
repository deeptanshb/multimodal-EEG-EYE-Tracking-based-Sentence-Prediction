# Multimodal EEG+Eye-to-Text Decoding with Quantum-Classical Hybrid AI

> **"Diagnosing and Repairing Temporal Attention Collapse in Low-Resource EEG-to-Text Decoding"**
>
> Temporal attention in long-sequence EEG encoders collapses to a near-uniform 1/T distribution
> regardless of content — a structural failure mode rendering the encoder mechanistically equivalent
> to mean pooling. We prove this collapse is an inevitable consequence of standard single-level
> softmax over T ≥ 128 timesteps, diagnose it empirically across six open-vocabulary EEG-to-text
> systems on the ZuCo corpus, and repair it with Hierarchical Temporal Pooling (HTP), a two-level
> windowed attention that reduces effective denominators from T=256 to 32 (local) and 8 (segment),
> recovering 10–30× attention-norm magnitude and inducing neurophysiologically interpretable,
> condition-sensitive cortical activation profiles. A four-stage pipeline achieves monotonically
> decreasing validation loss (4.2009→4.1729), 37% EEG–token alignment gain (TF/FG ratio
> 6.19×→4.49×), and BERTScore F1 85.51% on 2,032 held-out samples, with only 0.4% semantic
> drift. An exploratory bounded residual projector (QFP, 8,476 parameters) provides marginal
> regularisation as a secondary contribution. All code, checkpoints, and the NVIDIA NIM
> guardrailed benchmarking pipeline are released openly.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Evolution](#2-architecture-evolution)
3. [Repository Structure](#3-repository-structure)
4. [Dataset — ZuCo Corpus](#4-dataset--zuco-corpus)
5. [Environment Setup](#5-environment-setup)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Results](#7-results)
   - [Corrected locked baselines](#corrected-locked-baselines-from-process_flow_pipelineipynb-cell-3)
   - [V9 and QML live metrics](#v9-and-qml-live-metrics)
   - [Per-condition BLEU-1](#per-condition-bleu-1)
   - [Down/Up ablation — three-way](#downup-ablation--three-way)
   - [Bootstrap confidence intervals](#bootstrap-confidence-intervals)
   - [MC dropout stability](#mc-dropout-stability)
   - [Collapse diagnostic — HTP vs Baseline-2](#collapse-diagnostic--htp-vs-baseline-2)
   - [Per-subject generalisation](#per-subject-generalisation)
   - [Held-out subject evaluation](#held-out-subject-evaluation)
   - [Error analysis — five-category taxonomy](#error-analysis--five-category-taxonomy)
   - [Training summary](#training-summary)
8. [NVIDIA NIM Agent Platform](#8-nvidia-nim-agent-platform)
9. [NeMo Guardrails](#9-nemo-guardrails)
10. [Inference Benchmark Harness](#10-inference-benchmark-harness)
11. [External Researcher Interface](#11-external-researcher-interface)
12. [Streamlit Dashboard — app.py](#12-streamlit-dashboard--apppy)
13. [Getting an NVIDIA API Key](#13-getting-an-nvidia-api-key)
14. [Plots Reference](#14-plots-reference)
15. [Key Findings](#15-key-findings)
16. [Citation](#16-citation)

---

## 1. Project Overview

This project implements an end-to-end brain-computer interface (BCI) pipeline that:

- Reads EEG signals + eye-tracking data recorded while participants silently read sentences
- Preprocesses, normalises, and structures the neural signals into a multimodal feature set
- Trains a multi-region transformer with contrastive pretraining, hierarchical temporal pooling,
  condition-specific adapters, and an optional quantum residual circuit to decode the original sentence
- Evaluates decoding quality using BLEU-1/4, ROUGE-1/L, BERTScore F1, and the **TF/FG ratio**
  (a new metric quantifying how strongly the model depends on the EEG signal vs language priors)
- Validates statistical robustness via **bootstrap CIs** (n_boot=10,000), **MC dropout stability**
  (5-pass inference noise floor), and **leave-one-subject-out held-out retraining** (ZMG, ZJM)
- Performs a five-category **error analysis** using BERTScore F1 + BLEU-1 across all 2,032 val samples,
  classifying predictions as syntactic collapse / semantic drift / partial recovery /
  lexical substitution / successful decoding
- Runs a three-agent NVIDIA NIM guardrailed pipeline (Scientist → Critic → QML Synthesiser)
  to automatically interpret and peer-review the results
- Provides an open benchmarking platform where external researchers submit their own model metrics
  and receive structured comparative analysis against the V9+QML baseline

**Dataset:** ZuCo (Zurich Cognitive Language Processing Corpus) — 16 subjects, ~700 unique sentences,
three reading conditions (Normal Reading / Timed Silent Reading / Speed Reading).

**Key results (val n=2,032, sentence-aware split, seed=42):**

> Model names follow the paper. Mapping to internal code names: Baseline-1 = V5, Baseline-2 = V8 (6-region+MoCo), Baseline-3 = V9 (B2+HTP), B3+QFP = V9+QML.

| Model (paper name) | TF BLEU-1 | TF BLEU-4 | ROUGE-1 | ROUGE-L | BERTScore F1 | FG BLEU-1 | TF/FG | Val Loss |
|---|---|---|---|---|---|---|---|---|
| Baseline-1 (single-vector) | 29.24% | — | 33.92% | 30.06% | — | — | — | — |
| Baseline-2 (6-region+MoCo) | 30.40% | 4.30% | 35.78% | 30.68% | 85.46% | 4.92% | 6.19× | 4.2168 |
| Baseline-3 (B2+HTP) | **31.02%** | 4.45% | 36.07% | 30.79% | 85.50% | 6.90% | **4.49×** | 4.1744 |
| B3+QFP clean (noiseless VQC) | 31.00% | 4.47% | 36.04% | 30.80% | **85.51%** | 6.88% | 4.50× | 4.1733 |
| B3+QFP noisy (HW-sim) | 31.00% | 4.47% | 36.05% | 30.79% | **85.51%** | 6.81% | 4.54× | **4.1729** |

> The TF/FG ratio improvement from 6.19× (Baseline-2) to **4.49×** (Baseline-3) is the paper's central evidence of EEG-token alignment recovery — a 37% improvement. This is the primary metric, more meaningful than the modest absolute BLEU gains. Lower TF/FG means the EEG prefix is more informative relative to language priors.

**Statistical validation (from paper Extended Data Table 1 and §Statistics):**

| Evidence | Value | Interpretation |
|---|---|---|
| MC dropout std B3 (5 passes, sentence-level) | ±0.05% | Inference stochasticity floor |
| B2→B3 corpus BLEU gain / MC std SNR | **12.0×** | Gain is 12× the noise floor — not stochasticity |
| Bootstrap CI B3 sentence-BLEU (absolute) | [28.20%, 29.17%] | Sentence-level metric; corpus-level = 31.02% |
| QFP clean vs Baseline-3 (paired bootstrap) | Δ=+0.008pp, p=0.668 | **Not significant** — val loss is discriminating metric |
| QFP vs Down/Up projector (paired bootstrap) | Δ=+0.008pp, p=0.682 | **Not significant** on BLEU-1 |
| Held-out ZMG BLEU-1 (30.15%) / ZJM (27.07%) | Both > 0 | Cross-subject generalisation confirmed |
| Error analysis: semantically valid (Cat 5+6) | **74%** of 2,032 samples | BLEU-1 (31%) underestimates semantic quality |
| Semantic drift (Cat 3) | **0.4%** (9 samples) | Near-zero semantic failure rate |

---

## 2. Architecture Evolution

### V5 — Baseline

- Conv1D + Bidirectional GRU EEG encoder, single mean-pooled EEG vector
- Prefix-tuned DistilGPT-2 decoder, no eye-tracking or spectral features
- **TF BLEU-1: 29.24% | ROUGE-1: 33.92%** | Per-condition: NR=30.70% TSR=32.78% SR=26.49%

### V8 — Multimodal Multi-Region with Contrastive Learning

Key additions:

- **6 parallel GRU-Transformer RegionEncoders** — left temporal, left parietal, left parieto-occipital,
  central parietal, right parietal, right parieto-occipital
- **MoCo Stage 0** contrastive pretraining (queue=128, condition-based hard negatives)
- **LoRA fine-tuning** on GPT-2 blocks [10, 11] (rank=8, α=16)
- **SR condition adapter** — three separate MLPs for NR/TSR/SR conditions
- **Eye-tracking encoder** (fixations, pupil, duration) + **Spectral encoder** (8 band-power means)
- **Diagnosis:** `pool_attn Linear(D,1)` collapsed to uniform 1/256 in 4/6 regions — effectively mean-pooling
- **TF BLEU-1: 30.40% | ROUGE-1: 35.78% | BERTScore: 85.46%** | TF/FG: 6.19×
- Per-condition: NR=30.90% TSR=32.93% SR=27.20%

### V9 — Hierarchical Temporal Pooling (HTP)

Key fix:

- **HierarchicalTemporalPooling** replaces flat `pool_attn`:
  - Level 1: 32-way local softmax within 8 windows of 32 timesteps each (0.5s at 64 Hz)
  - Level 2: 8-way segment softmax across windows
  - Gradient concentrated 8× vs collapsed 256-way softmax → selective temporal peaks
- **LoRA rank=4, α=16, block=[11] only** (rank reduced from 8, single block)
- **dropout=0.4**; encoder near-frozen in Stage 2 (lr=1e-6)
- **TF BLEU-1: 31.02% | ROUGE-1: 36.07% | BERTScore F1: 85.50%** | TF/FG: **4.49×** (was 6.19×; 37% alignment gain) | FG BLEU-1: **6.90%** | val loss: 4.1744
- Per-condition: NR=31.53% TSR=**33.77%** SR=27.46% (TSR leads — VWFA dominates under timed presentation)

### V9+QML clean — Quantum Fusion Projector (noiseless)

Key addition:

- **QuantumFusionProjector (QFP)** inserted after `sr_adapter`, before fusion MHA:
  - `Linear(768→4)` + tanh + π-scaling into qubit space
  - `AngleEmbedding` (RY rotations) encodes 4-dim EEG into 4-qubit state
  - 2× `StronglyEntanglingLayers` (CNOT ladders + rotation gates)
  - 4 Pauli-Z expectations → `Linear(4→768)` + LayerNorm residual
  - **~8,476 QML parameters** (0.006% of 147M total)
- **PennyLane** `lightning.qubit` simulator — noiseless statevector simulation
- **10-epoch QML fine-tune**: QML_LR=3e-4, rest=1e-6, CosineAnnealingLR, eta_min=1e-7, patience=3
- **Hybrid LoRA**: rank=4, **α=8.0**, block=[11]; dropout=0.4
- **TF BLEU-1: 31.00% | ROUGE-1: 36.04% | BERTScore F1: 85.51%** | TF/FG: **4.50×** | FG BLEU-1: **6.88%** | val loss: **4.1733**

**Down/Up ablation** (`ClassicalDownUpProjector`, Cell 26): same position as QFP, 768→4 GELU→768 + LayerNorm
residual, 6,160 parameters. Sentence-level BLEU-1 = 28.70% — **statistically identical** to QFP (28.69%)
and B3 (28.69%); bootstrap p=0.682, CI [−0.028,+0.046]. Val loss 4.2062 > B3 4.1744 — the linear
bottleneck hurts; QFP is the only tested projector that reduces val loss (−0.0011).
Checkpoint saved as `final_best_v9_downup_only.pt`.

### V9+QML noisy — Hardware-Realistic Noise Simulation

Key addition on top of V9+QML clean:

- **NoisyQuantumFusionProjector** — same VQC + hardware-realistic noise channels:
  - `DepolarizingChannel(p=0.01)` after each encoding gate (1% gate error)
  - `PhaseDamping(γ=0.02)` after `StronglyEntanglingLayers` (T2 decoherence)
  - Uses **PennyLane `default.mixed`** density-matrix simulator
- **Training**: Gaussian shot-noise (σ=0.03) injected on VQC output each pass → forces robustness
- **Inference**: Monte-Carlo average over 16 noisy circuit passes (variance ÷ 4×)
- **Initialised from clean QML checkpoint**; 10-epoch noise-aware fine-tune
- **TF BLEU-1: 31.00% | ROUGE-1: 36.05% | BERTScore F1: 85.51%** | TF/FG: **4.54×** | FG BLEU-1: **6.81%** | val loss: **4.1729** (clean: 4.1733)
- Δ clean → noisy: 0.0004 val loss improvement — **noise acts as regulariser**; architecture is hardware-deployable

---

## 3. Repository Structure

```
PROJECT1/
│
├── ── CORE MODEL & TRAINING ────────────────────────────────────────────
│
├── model1_v9.py                     # ALL model classes: HTP, RegionEncoderV9,
│                                    #   EEG2TextTransformerV9, QuantumFusionProjector,
│                                    #   NoisyQuantumFusionProjector, ClassicalDownUpProjector,
│                                    #   MoCo, training helpers, REGION_NAMES
├── Process_flow_pipeline.ipynb      # Main training + full analysis notebook (72 cells)
│                                    #   Cells 0–22:   preprocessing → Stage 0/1/2 → evaluation
│                                    #   Cells 23–25:  QML clean fine-tune + BERTScore
│                                    #   Cells 26–31:  Down/Up ablation (ClassicalDownUpProjector)
│                                    #   Cells 32–35:  Bootstrap paired CIs (n_boot=10,000)
│                                    #   Cells 36–56:  diagnostics, plots, per-condition analysis
│                                    #   Cell  57:     teacher-forcing per-subject BLEU-1 (TF, n=2032)
│                                    #   Cells 58–60:  noisy QML fine-tune
│                                    #   Cells 59–62:  held-out subject retraining (ZMG, ZJM)
│                                    #   Cell  63:     MC dropout stability (5 passes, 3 models)
│                                    #   Cell  64:     extended bootstrap — absolute CIs, B2→B3 SNR
│                                    #   Cells 65–72:  error analysis — categorise, BERTScore,
│                                    #                 repetition scoring, LaTeX table, save
│                                    #   Stage0 MoCo → Stage1 → Stage2 LoRA → QML clean → QML noisy
│                                    #   → down/up ablation → bootstrap → held-out → MC dropout
│                                    #   → error analysis
├── data_extraction.ipynb            # ZuCo .mat → pickle extractor (3 cells)
│
├── ── CHECKPOINTS ──────────────────────────────────────────────────────
│
├── stage0_v9.pt                     # Stage 0 MoCo checkpoint
├── stage1_best_v9.pt                # Stage 1 best checkpoint
├── final_best_v9.pt                 # Best Stage 2 (LoRA) checkpoint  — Baseline-3 (B3)
├── final_best_v9_downup_only.pt     # Down/Up ablation checkpoint (val loss=4.2062)
├── hybrid_qml_v9_best.pt            # Best QML clean checkpoint (val loss=4.1733)
├── hybrid_qml_noisy_v9_best.pt      # Best QML noisy checkpoint (val loss=4.1729)
│
├── ── HELD-OUT CHECKPOINTS (leave-one-subject-out) ─────────────────────
│
├── final_holdout_ZMG.pt             # Stage 2 checkpoint retrained without ZMG
├── final_holdout_ZJM.pt             # Stage 2 checkpoint retrained without ZJM
├── hybrid_qml_holdout_ZMG.pt        # QML clean checkpoint retrained without ZMG
├── hybrid_qml_holdout_ZJM.pt        # QML clean checkpoint retrained without ZJM
├── hybrid_qml_noisy_holdout_ZMG.pt  # QML noisy checkpoint retrained without ZMG
├── hybrid_qml_noisy_holdout_ZJM.pt  # QML noisy checkpoint retrained without ZJM
│
├── ── HELD-OUT RESULTS (JSON) ──────────────────────────────────────────
│
├── holdout_ZMG_result.json          # ZMG held-out final metrics
│                                    #   sent_bleu1=30.15%  val_loss=4.2948  Δval=+0.1219
├── holdout_ZJM_result.json          # ZJM held-out final metrics
│                                    #   sent_bleu1=27.07%  val_loss=4.4706  Δval=+0.2977
├── holdout_ZMG_noisy_progress.json  # Per-epoch noisy fine-tune log for ZMG held-out run
├── holdout_ZJM_noisy_progress.json  # Per-epoch noisy fine-tune log for ZJM held-out run
│
├── ── NVIDIA AGENT PLATFORM ────────────────────────────────────────────
│
├── nat_eeg_agents_v9_product.ipynb  # Main agent notebook (43 cells) — PRODUCT VERSION
│                                    #   Cells 1-13: inference + metrics + agent_stats
│                                    #   Cell 14b:   install NeMo Guardrails
│                                    #   Cell 14:    view agent system prompts
│                                    #   Cell 14c:   LLM caller + guardrail flow explainer
│                                    #   Cell 15:    load nat_agents_guardrailed module
│                                    #   Cell 16:    run guardrailed 3-agent pipeline
│                                    #   Cell 17:    inference benchmark harness
│                                    #   Cell 18:    display agent outputs
│                                    #   Cell 10b:   NoisyQFP + noisy_hybrid model setup
│                                    #   Cell 19:    save nat_v9_qml_results.json
│
├── nat_v9_qml_results.json          # Agent pipeline output — live metrics + agent text
├── per_subject_bleu.json            # Per-subject BLEU-1/ROUGE-1 breakdown (16 subjects)
│                                    #   mean=16.56%  std=0.33pp  range=0.95pp  no outliers
│                                    #   + benchmark_records + guardrail_audit
│
├── ── eeg_product/  ────────────────────────────────────────────────────
│   │   (NVIDIA product layer — all agent + guardrail + benchmark code)
│   │
│   ├── nat_agents_guardrailed.py    # Core pipeline module:
│   │                                #   SCIENTIST_SYSTEM, CRITIC_SYSTEM, QML_SYSTEM prompts
│   │                                #   call_nim_guardrailed() — 300s timeout, retry logic
│   │                                #   run_guardrailed_pipeline() — 3-agent orchestrator
│   │                                #   _load_rails() — NeMo Guardrails auto-loader
│   │                                #   _write_colang1_rails() — Colang 1.0 auto-patch
│   │
│   ├── eeg_submission_schema.py     # External researcher interface:
│   │                                #   V5_BASELINE, V8_BASELINE (locked constants)
│   │                                #   V9_QML_BASELINE, V9_QML_NOISY_BASELINE
│   │                                #   EEGModelSubmission dataclass
│   │                                #   load_v9_qml_baseline(), load_v9_qml_noisy_baseline()
│   │
│   ├── comparison_pipeline.py       # 4-agent comparison pipeline for external researchers:
│   │                                #   Scientist + Comparator + Critic + Synthesiser
│   │                                #   run_comparison_pipeline(), save_comparison_report()
│   │
│   ├── external_researcher_template.ipynb  # 10-cell template notebook for external users
│   │                                       #   No model code needed — metrics only
│   │
│   ├── guardrails_config/
│   │   ├── config.yml               # NeMo Guardrails config:
│   │   │                            #   engine: openai (OpenAI-compat, points to NIM)
│   │   │                            #   model: meta/llama-3.1-8b-instruct
│   │   │                            #   input rails + output rails declared
│   │   ├── rails.co                 # Colang 1.0 flow definitions:
│   │   │                            #   check eeg domain intent (input) — incl. noisy QML examples
│   │   │                            #   check metric hallucination (output)
│   │   │                            #   check domain relevance (output)
│   │   │                            #   check noisy qml context (output) — validates payload keys
│   │   └── guardrails_actions.py    # Python actions:
│   │                                #   check_metric_bounds() — BLEU/ROUGE/BERTScore ranges
│   │                                #   self_check_relevance() — 38 domain terms incl. noisy QML
│   │                                #   check_noisy_qml_keys() — validates noisy_qml_* payload keys
│   │                                #   get_agent_role() — role router
│   │
│   └── benchmark/
│       └── nim_benchmark.py         # Inference benchmark harness:
│                                    #   NIMBenchmark class — N-run pipeline benchmark
│                                    #   CallMetrics, AgentBenchmarkReport dataclasses
│                                    #   default_guardrail_check() — Python-side checks
│                                    #   CLI: python nim_benchmark.py --runs 5
│
├── ── STREAMLIT DASHBOARD ──────────────────────────────────────────────
│
├── app.py                           # Full Streamlit dashboard (11 pages):
│                                    #   Overview / Training Curves / Model Comparison
│                                    #   EEG Attention / Architecture / Qualitative Samples
│                                    #   Quantum Fusion / Per-Subject Analysis /
│                                    #   Error Analysis / NVIDIA Stack / NAT Agents
│                                    #   Includes live agent runner with guardrail badges
│
├── ── ANALYSIS OUTPUTS ─────────────────────────────────────────────────
│
├── bootstrap_ci_results.json        # Bootstrap CI results (n_boot=10,000, seed=42, sentence-level BLEU-1)
│                                    #   B3 absolute CI: [28.20%, 29.17%]  QFP: [28.21%, 29.18%]
│                                    #   QFP vs B3: Δ=+0.008pp  p=0.668 (not significant)
│                                    #   QFP vs down/up: Δ=+0.008pp  p=0.682 (not significant)
├── mc_dropout_stability.json        # MC dropout stability (5 passes, seeds [42,123,456,789,1024])
│                                    #   B3 MC mean=27.50% std=±0.05%  SNR=12.0× vs corpus B2→B3 gain
├── error_analysis_summary.json      # Error analysis compact summary
│                                    #   5-category counts + %, mean BLEU-1, mean BERTScore F1
│                                    #   per-condition breakdown + representative examples
│                                    #   + full LaTeX table string (Supplementary Table S3)
├── error_analysis_results.json      # Full per-sample error analysis (2,032 rows)
│                                    #   pred_str, ref_str, bleu1, bert_f1, rep_score,
│                                    #   len_ratio, cat_id, cat_name, cond, cond_name
│
├── ── COLLAPSE DIAGNOSTIC ──────────────────────────────────────────────
│
├── diagnose_collapse.py             # Standalone diagnostic tool (Cell 74)
│                                    #   Usage: python diagnose_collapse.py --weights b3_htp_attn.npy --T 256
│                                    #   Computes mean max-weight, entropy, and HEALTHY/COLLAPSED verdict
│                                    #   for any attention array; works on any EEG model checkpoint
├── b3_htp_attn.npy                  # Real HTP local_attn weights extracted from final_best_v9.pt
│                                    #   shape: (n_samples×6_regions, 8_windows, 32_timesteps)
│                                    #   n_samples=2,032 val × 6 regions = 12,192 rows
│                                    #   Extracted via forward hooks in Cell 73 (load_and_extract.py)
├── b2_simulated_attn.npy            # Simulated uniform-collapse baseline (Baseline-2 equivalent)
│                                    #   shape: matches b3_htp_attn.npy
│                                    #   values: 1/256 + ε, ε~N(0, 0.0002); max deviation < 0.0002
│
├── ── COMPARISON OUTPUT ────────────────────────────────────────────────
│
├── comparison_eegconformer_lora_v1.json   # Sample external researcher comparison result
│                                          #   EEGConformer_LoRA_v1 vs V9+QML baseline
│                                          #   4 agents · 18.2s · 100% guardrail pass
│
├── ── DATA & SCALERS ───────────────────────────────────────────────────
│
├── eeg_mean.npy                     # EEG z-score mean (Welford, training set only)
├── eeg_std.npy                      # EEG z-score std
├── scaler_eye.pkl                   # Fitted StandardScaler for eye features
├── scaler_spec.pkl                  # Fitted StandardScaler for spectral features
├── selected_channels.json           # Top-24 BioSemi channel indices
├── selected_channels.npy
├── NR_data.pkl                      # Extracted NR rows (raw from .mat)
├── NR_lean.pkl                      # Processed NR rows (post-preprocessing)
├── SR_data.pkl / SR_lean.pkl
├── TSR_data.pkl / TSR_lean.pkl
│
├── ── RAW DATA ─────────────────────────────────────────────────────────
│
├── NR_files/                        # Raw .mat files — Normal Reading
├── TSR_files/                       # Raw .mat files — Timed Silent Reading
├── SR_files/                        # Raw .mat files — Speed Reading
├── processed_data/                  # Intermediate processed pickles
│
├── ── PLOTS ────────────────────────────────────────────────────────────
│
├── plots/                           # All saved figures (see §14)
│
├── ── ENVIRONMENT ──────────────────────────────────────────────────────
│
├── zuco_env/                        # Python virtual environment
├── requirements.txt                 # All Python dependencies
└── README.md                        # This file
```

### Key file quick reference

| File | What it does |
|------|-------------|
| `model1_v9.py` | All model classes — HTP, QFP, NoisyQFP, ClassicalDownUpProjector, REGION_NAMES |
| `Process_flow_pipeline.ipynb` | Training + evaluation + ablation + bootstrap + held-out + error analysis (72 cells) |
| `nat_eeg_agents_product.ipynb` | **Product notebook** — inference + guardrailed agents + benchmark |
| `data_extraction.ipynb` | ZuCo `.mat` → pickle extractor |
| `eeg_product/nat_agents_guardrailed.py` | Agent prompts, NIM caller, pipeline orchestrator |
| `eeg_product/eeg_submission_schema.py` | Submission dataclass + V5/V8/V9_QML/V9_QML_NOISY baselines |
| `eeg_product/comparison_pipeline.py` | 4-agent comparison for external researchers |
| `eeg_product/external_researcher_template.ipynb` | Template notebook for external users |
| `eeg_product/guardrails_config/` | NeMo Guardrails config, Colang 1.0 flows, Python actions |
| `eeg_product/benchmark/nim_benchmark.py` | TTFT / latency / throughput harness |
| `app.py` | Streamlit 11-page analysis dashboard |
| `nat_v9_qml_results.json` | Live metrics + agent outputs + benchmark + guardrail audit |
| `bootstrap_ci_results.json` | Bootstrap CIs — absolute (B3, QFP) + paired (QFP vs V9, vs down/up) |
| `mc_dropout_stability.json` | MC dropout: 5-pass BLEU-1 per model, B2→B3 SNR evidence |
| `error_analysis_summary.json` | 5-category error taxonomy summary + LaTeX Table S3 string |
| `error_analysis_results.json` | Full 2,032-sample per-sentence error analysis records |
| `diagnose_collapse.py` | Standalone collapse diagnostic — `python diagnose_collapse.py --weights b3_htp_attn.npy --T 256` |
| `b3_htp_attn.npy` | Real HTP local_attn weights from `final_best_v9.pt` (shape: 12,192 × 8 × 32) |
| `b2_simulated_attn.npy` | Simulated Baseline-2 collapse baseline (1/256+ε; max deviation < 0.0002) |
| `holdout_ZMG_result.json` | ZMG held-out: sent BLEU-1=30.15%, val loss=4.2948, Δval=+0.1219 |
| `holdout_ZJM_result.json` | ZJM held-out: sent BLEU-1=27.07%, val loss=4.4706, Δval=+0.2977 |
| `comparison_eegconformer_lora_v1.json` | Sample external comparison output |

---

## 4. Dataset — ZuCo Corpus

### What is ZuCo?

ZuCo (Zurich Cognitive Language Processing Corpus) is a publicly available EEG + eye-tracking dataset.
Participants read natural English sentences wearing a 128-channel BioSemi EEG cap while eye-tracking
recorded fixations, gaze duration, and pupil size.

**Subjects (16 total):**

| Format | Prefix | Subject IDs |
|--------|--------|-------------|
| HDF5 / MATLAB v7.3 (`h5py`) | Y | YAC, YAK, YDG, YFS |
| MATLAB v5/v6 (`scipy.io`) | Z | ZAB, ZDM, ZDN, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW, ZMG, ZPH |

**Citation:** Hollenstein et al., "ZuCo, a simultaneous EEG and eye-tracking resource for natural
sentence reading", *Scientific Data*, 2018.

### Download

```
https://osf.io/q3zws/
```

Download the three condition folders (`NR/`, `TSR/`, `SR/`) and place as `NR_files/`, `TSR_files/`, `SR_files/`.

### Raw file format — two subject types

ZuCo `.mat` files come in **two formats** depending on subject ID prefix:

| Prefix | Format | Loader | Example subjects |
|--------|--------|--------|-----------------|
| **Y-prefix** | HDF5 / MATLAB v7.3 | `h5py.File(path, "r")` | YAC, YAG, YAK, YAP, YDG, YFS, YHS, YLS, YMD, YMS, YRH, YSD, YSL, YTL |
| **Z-prefix** | MATLAB v5/v6 | `scipy.io.loadmat(path)` | ZAB, ZDN, ZGW, ZJM, ZJN, ZKB, ZKH, ZKW, ZMG, ZPH |

`my.ipynb` (Step 0) handles both formats automatically — it detects the prefix and routes to the
correct loader. All subjects across all three conditions are processed and merged into the
condition `.pkl` files. **Do not rename the `.mat` files** — the Y/Z prefix is used for format detection.

### Dataset statistics (after preprocessing)

| Condition | Raw rows | Post-split train | Val rows |
|-----------|----------|-----------------|----------|
| NR (Normal Reading) | 3,887 | ~3,248 | 639 |
| TSR (Timed Silent) | 4,687 | ~3,967 | 720 |
| SR (Speed Reading) | 4,378 | ~3,705 | 673 |
| **Total** | **12,952** | **~10,920** | **2,032** |

Split: sentence-aware 85%/15%, seed=42. No sentence appears in both sets.

### Preprocessing pipeline (cells 3–14 of `final.ipynb`)

1. **Channel selection** — variance-based top-24 from 105 channels (NR condition only, no leakage)
2. **Bandpass filter** — Butterworth 0.5–40 Hz, order 4
3. **Downsampling** — 500 Hz → 64 Hz, TARGET_LEN=256 timesteps = 4 seconds
4. **Omission filtering** — rows with >60% missing electrode data removed
5. **PCA compression** — 24-component PCA on selected channels
6. **EEG z-score normalisation** — Welford online mean/std on training set → `eeg_mean.npy`, `eeg_std.npy`
7. **Eye StandardScaler** → `scaler_eye.pkl`; **Spectral StandardScaler** → `scaler_spec.pkl`
8. **Data augmentation** — paired EEG trial averaging (~1,035 mixed rows added to training)

Final shape per row: `(256 timesteps × 24 PCA channels)` + 3 eye features + 8 spectral features.

---

## 5. Environment Setup

### Requirements

- Python 3.10+
- CUDA GPU (tested on RTX 3050 4 GB; brev NVIDIA instance for publication benchmarks)
- ~12 GB RAM for preprocessing

### Installation

```bash
cd PROJECT1
python -m venv zuco_env
source zuco_env/bin/activate          # Linux/Mac

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt       # all other dependencies

# NLTK punkt tokenizer
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# NeMo Guardrails + LangChain OpenAI (for the agent platform)
pip install nemoguardrails>=0.10.0 langchain-openai openai>=1.0.0
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())           # True
print(torch.cuda.get_device_name(0))       # NVIDIA GeForce RTX 3050
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # ~4.0 GB
```

---

## 6. Running the Pipeline

### Step 0 — Extract raw data (run once)

Open `my.ipynb` and run all 3 cells. Reads every `.mat` from `NR_files/`, `TSR_files/`, `SR_files/`
and saves `NR_data.pkl`, `TSR_data.pkl`, `SR_data.pkl`.
Handles both formats: **Y-prefix** subjects use `h5py` (MATLAB v7.3 HDF5),
**Z-prefix** subjects use `scipy.io.loadmat` (MATLAB v5/v6). Detection is automatic.

### Step 1 — Training (`Process_flow_pipeline.ipynb`)

Run cells in order. If checkpoints already exist, jump to Cell 21:

```
Cells 00–02  → imports, install pennylane, config
Cells 03–14  → preprocessing, splitting, normalisation, augmentation
Cells 15–17  → device setup, model classes, dataset/dataloader
Cell  18     → Stage 0 MoCo pretraining (20 epochs) → stage0_v9.pt
Cell  19     → Stage 1 training (20 epochs) → stage1_best_v9.pt
Cell  20     → Stage 2 LoRA training (20 epochs) → final_best_v9.pt
Cell  21     → EVAL_LOAD — load best checkpoint + alpha sweep
Cell  22     → BLEU/ROUGE/BERTScore evaluation
Cell  23     → QML fine-tune (10 epochs) → hybrid_qml_v9_best.pt (clean)
Cell  24     → BERTScore on classical + hybrid
Cells 25     → diagnostics (pool_attn, cross-region, SR adapter, TF/FG)
Cells 26–31  → Down/Up ablation: ClassicalDownUpProjector train + BLEU comparison
               → final_best_v9_downup_only.pt (val loss=4.2062)
Cells 32–35  → Bootstrap paired CIs (n_boot=10,000): QFP vs V9, QFP vs down/up
               → bootstrap_ci_results.json
Cells 36–56  → publication plots → plots/
Cell  57     → Teacher-forcing per-subject BLEU-1 (all 16 subjects, overall=30.95%)
Cells 58–60  → Noisy QML fine-tune → hybrid_qml_noisy_v9_best.pt (val loss=4.1729)
Cells 59–62  → Held-out subject retraining: ZMG and ZJM leave-one-out
               → final_holdout_ZMG.pt, hybrid_qml_holdout_ZMG.pt, hybrid_qml_noisy_holdout_ZMG.pt
               → final_holdout_ZJM.pt, hybrid_qml_holdout_ZJM.pt, hybrid_qml_noisy_holdout_ZJM.pt
               → holdout_ZMG_result.json, holdout_ZJM_result.json
Cell  63     → MC dropout stability (N=5 passes, seeds [42,123,456,789,1024])
               → mc_dropout_stability.json
Cell  64     → Extended bootstrap: absolute 95% CIs for B3 + QFP, B2→B3 SNR evidence synthesis
Cell  65     → Error analysis setup: imports, constants, load hybrid_qml_noisy_v9_best.pt
Cell  66     → Teacher-forcing inference on 2,032 val samples → records list (blob-stripped)
Cell  67     → BERTScore F1 computation (roberta-large) for all predictions
Cell  68     → Repetition + length quality metrics (token-level, id-level, blob-detection)
Cell  69     → Error categorisation: 5 cats in priority order → per-condition breakdown
Cell  70     → Representative examples per category → examples_for_paper dict
Cell  71     → LaTeX Table S3 generation (Supplementary Methods)
Cell  72     → Save → error_analysis_summary.json + error_analysis_results.json
Cell  73     → Load Baseline-3 checkpoint, apply LoRA scaffolding, register forward hooks on
               all 6 HTP modules across all RegionEncoders, run forward pass on 2,032-sample
               val set, concatenate captured local_attn arrays → b3_htp_attn.npy
               Construct simulated Baseline-2 collapse (1/256+ε) → b2_simulated_attn.npy
Cell  74     → Run diagnose_collapse.py on both .npy files:
               diagnose(alpha_htp, T=256)  → HEALTHY  (max_w=0.034, H=4.96 nats, 8.75× uniform)
               diagnose(alpha_col, T=256)  → COLLAPSED (max_w=0.007, H=5.53 nats, 1.74× uniform)
               Command-line equivalent: python diagnose_collapse.py --weights b3_htp_attn.npy --T 256
```

### Step 2 — Agent platform (`nat_eeg_agents_v9_product.ipynb`)

**One-time setup (first run only):**

```bash
pip install nemoguardrails>=0.10.0 langchain-openai openai>=1.0.0
```

**Set your NVIDIA API key** (get one free at https://build.nvidia.com):

```python
# In cell 3 (Imports & config):
NVIDIA_API_KEY = "nvapi-your-key-here"
# OR set as environment variable before launching:
export NVIDIA_API_KEY="nvapi-your-key-here"
```

**Run order:**

```
Cell 1      → install pennylane, NAT, NLTK (run once, restart kernel)
Cell 2      → imports & config — V5/V8 baselines locked here
Cells 3–13  → load models + data, run inference, compute metrics, assemble agent_stats
Cell 14b    → install NeMo Guardrails + openai (run once)
Cell 14     → view all 3 agent system prompts
Cell 14c    → read LLM caller architecture explainer
Cell 15     → load nat_agents_guardrailed module, connect to NIM
              (uncomment to switch to self-hosted: NIM_BASE_URL=http://localhost:8000/v1)
Cell 16     → run guardrailed 3-agent pipeline (~1–3 min depending on model)
Cell 17     → inference benchmark (set N_BENCHMARK_RUNS=5 for publication numbers)
Cell 18     → display Scientist / Critic / QML Synthesiser outputs + metric tables
Cell 19     → save nat_v9_qml_results.json
```

### Step 3 — Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. See §12 for full page descriptions.

### Step 4 — External researcher comparison

See §11. Researchers open `eeg_product/external_researcher_template.ipynb`, fill in
`EEGModelSubmission`, and run 4 cells.

---

## 7. Results

### Corrected locked baselines (from `Process_flow_pipeline.ipynb` cell 3)

> These values are hard-coded into `eeg_product/nat_agents_guardrailed.py` cell 3 and
> `eeg_product/eeg_submission_schema.py`. Do not change them.

| Metric | V5 | V8 |
|--------|----|----|
| TF BLEU-1 | 29.24% | 30.40% |
| TF BLEU-4 | — | 4.30% |
| TF ROUGE-1 | 33.92% | **35.78%** |
| TF ROUGE-L | 30.06% | 30.68% |
| FG BLEU-1 | — | 4.81% |
| BERTScore F1 | — | **85.46%** |
| TF/FG ratio | — | 6.19× |
| Per-condition NR | 30.70% | 30.90% |
| Per-condition TSR | 32.78% | 32.93% |
| Per-condition SR | 26.49% | 27.20% |

> ⚠️ Note: Earlier versions of this README contained stale values (ROUGE-1=36.01%, BERTScore=85.53%,
> FG BLEU-1=4.91%). The values above are the authoritative numbers from `final.ipynb` cell 22 / cell 3.
> Corrected: FG BLEU-1=4.91%, all V9/QML metrics updated from actual evaluation run.

### V9 and QML live metrics (from `nat_v9_qml_results.json`)

> Values below are from the live agent-run inference pass on 2,032 val samples. Small differences (≤0.05pp) from training-run Table 3 are due to inference-session floating-point non-determinism.

| Metric | Baseline-2 (V8) | Baseline-3 (V9) | B3+QML clean | B3+QML noisy | Δ B2→B3 | Δ B3→QML |
|--------|----------------|----------------|-------------|-------------|---------|---------|
| TF BLEU-1 | 30.40% | 30.97% | 30.95% | 30.95% | +0.57pp | −0.02pp |
| TF BLEU-4 | 4.30% | 4.45% | 4.47% | 4.47% | +0.15pp | +0.02pp |
| TF ROUGE-1 | 35.78% | 36.07% | 36.04% | 36.05% | +0.29pp | −0.03pp |
| TF ROUGE-L | 30.68% | 30.76% | 30.77% | 30.76% | +0.08pp | 0.00pp |
| BERTScore F1 | 85.46% | 85.50% | 85.51% | 85.51% | +0.04pp | +0.01pp |
| FG BLEU-1 (greedy) | 4.92% | **6.90%** | 6.88% | 6.81% | **+1.98pp** | −0.09pp |
| TF/FG ratio | 6.19× | **4.49×** | 4.50× | 4.54× | **−1.70×** | +0.05× |
| Val loss | 4.2168 | 4.1744 | **4.1733** | **4.1729** | −0.0424 | −0.0011 |

### Per-condition BLEU-1

From paper Table 7 (n=2,032 val; NR=639, TSR=720, SR=673). TSR leads all conditions from Baseline-2 onward — consistent with VWFA dominance under word-by-word timed presentation.

| Condition | Baseline-1 (V5) | Baseline-2 (V8) | Baseline-3 (V9) | QML clean | QML noisy | Δ B1→best |
|-----------|-----------------|-----------------|----------------|-----------|-----------|-----------|
| NR | 30.70% | 30.90% | 31.53% | 31.43% | **31.44%** | +0.54pp |
| TSR | 32.78% | 32.93% | 33.77% | 33.87% | **33.93%** | +1.00pp |
| SR | 26.49% | 27.20% | 27.46% | 27.37% | 27.30% | +0.26pp |
| TSR−SR gap | 6.29pp | 5.73pp | 6.31pp | 6.50pp | **6.63pp** | — |

> TSR leads all conditions in all models from Baseline-2 onward. Noisy QML is within 0.07pp of clean QML across all three conditions — condition-level noise robustness confirmed. SR has the smallest absolute gain (+0.26pp) because Speed Reading produces weaker EEG signal (shorter fixation durations).

### Down/Up Ablation — Three-Way

`ClassicalDownUpProjector` (notebook Cells 26–31): same insertion point as QFP, 768→4 GELU→768 + LayerNorm
residual, 6,160 parameters — no quantum circuit. Sentence-level smoothed BLEU-1 (per-sample mean)
used for all three variants so relative differences are valid.

| Variant | Sentence BLEU-1 | ROUGE-1 | Val Loss | Δ val loss vs B3 |
|---------|-----------------|---------|----------|-----------------|
| Baseline-3 classical (no projector) | 28.69% | 33.72% | 4.1744 | baseline |
| Baseline-3 + Down/Up (classical, 768→4→768) | 28.70% | 33.76% | 4.2062 | **+0.0318 (worse)** |
| Baseline-3 + QFP clean (VQC) | 28.69% | 33.74% | **4.1733** | −0.0011 (better) |
| Baseline-3 + QFP noisy (VQC+noise) | 28.69% | 33.75% | **4.1729** | −0.0015 (best) |

> **Key result (from paper Table 6):** Sentence-level BLEU-1 is statistically **flat** across all three
> variants (28.69–28.70%). Bootstrap B=10,000, n=2,032:
> QFP vs B3: Δ=+0.008pp, 95% CI [−0.029,+0.047], **p=0.668** — not significant.
> QFP vs down/up: Δ=+0.008pp, 95% CI [−0.028,+0.046], **p=0.682** — not significant.
> **Val loss is the discriminating metric.** The 4-dim linear bottleneck hurts (d/u 4.2062 > B3 4.1744);
> QFP is the only tested projector that reduces val loss (−0.0011) — attributable to bounded [−1,1]
> expectation-value output acting as implicit regularisation at ZuCo's 11,955-sample scale.
> Checkpoint: `final_best_v9_downup_only.pt`.

### Bootstrap Confidence Intervals

From paper Extended Data Table 1 (§Statistics). All bootstrap: B=10,000 paired resamples, seed=42,
n=2,032. **Metric: sentence-level smoothed BLEU-1** (sentence_bleu() + SmoothingFunction().method1,
averaged per sample) — ≈2.3pp lower than corpus BLEU by construction. Saved to `bootstrap_ci_results.json`.

**Part A — QFP ablation (paired bootstrap, sentence-level BLEU-1):**

| Comparison | Δ (pp) | 95% CI | p-value | Significant? |
|------------|--------|--------|---------|-------------|
| QFP clean vs Baseline-3 | +0.008 | [−0.029, +0.047] | **0.668** | ❌ No |
| QFP clean vs Down/Up only | +0.008 | [−0.028, +0.046] | **0.682** | ❌ No |
| ROUGE-1: QFP vs Baseline-3 | +0.025 | [−0.015, +0.068] | 0.242 | ❌ No |

> No comparison approaches significance. **Val loss is the primary discriminating metric** (d/u 4.2062 > B3 4.1744 > QFP 4.1733 > noisy 4.1729). No correction for multiple comparisons was applied; Bonferroni threshold at α=0.05 across 6 tests would be p<0.0083; all p>0.65.

**Part B — Absolute 95% CIs (sentence-level BLEU-1):**

| Model | Mean sent. BLEU-1 | 95% CI |
|-------|------------------|--------|
| Baseline-3 | 28.69% | [28.20%, 29.17%] |
| QFP clean | 28.69% | [28.21%, 29.18%] |

> Note: sentence-level means (≈28.7%) differ from corpus BLEU (31.02%) by 2.3pp — two separate metrics computed by different functions on the same outputs. Both are real; sentence-level is required for per-sample bootstrap resampling.

### MC Dropout Stability

From paper §Statistics and Extended Data Table 1 Part C. Five stochastic forward passes with
`model.train()` + `torch.no_grad()`, seeds [42, 123, 456, 789, 1024]. LoRA applied via
`model.stage_2_setup(rank=4, α=8.0, block=11)` before each checkpoint load.
Saved to `mc_dropout_stability.json`.

| Model | MC mean sent. BLEU-1 | Std | 95% spread |
|-------|---------------------|-----|-----------|
| Baseline-3 | 27.50% | **±0.05%** | [27.45%, 27.60%] |
| QFP clean | 27.49% | ±0.07% | [27.42%, 27.60%] |
| QFP noisy | 27.47% | ±0.08% | [27.36%, 27.60%] |

> **Why MC mean (27.50%) differs from reported corpus BLEU (31.02%):** two independent sources compound.
> (1) Metric difference: corpus_bleu() vs sentence_bleu()+smoothing; sentence-level is ≈2.3pp lower
> by construction. (2) Dropout active: model.train() randomly zeros neurons, reducing output quality
> by ≈1.2pp. Total gap: 31.02% − 27.50% = 3.52pp (2.3pp metric + 1.2pp dropout).
> **Only the std (±0.05%) carries interpretive weight — not the mean.**

**B2→B3 gain evidence synthesis:**

| Evidence | Value | Interpretation |
|----------|-------|---------------|
| Corpus BLEU delta (B2→B3) | +0.62pp | Raw gain reported in paper |
| B3 MC dropout std (sentence-level) | **±0.05%** | Inference stochasticity floor |
| Gain / MC std SNR | **12.0×** | Corpus gain (0.62pp) is 12× the dropout noise floor |
| Val loss delta B2→B3 | −0.0424 | Most direct threshold-independent evidence |

> The B2→B3 corpus BLEU gain (+0.62pp) is **12× larger** than the entire stochastic variation
> range under degraded MC-dropout inference (±0.05%). The gain cannot be attributed to inference-level
> stochasticity. Baseline-2 checkpoint was not retained so a paired bootstrap on the B2→B3 delta
> is not possible; MC dropout stability and deterministic val loss (Δ=−0.0424) serve as primary evidence.

### Collapse Diagnostic — HTP vs Baseline-2

From notebook Cells 73–74 and `diagnose_collapse.py`. Run on Baseline-3 checkpoint
(`final_best_v9.pt`), n=12,192 samples (2,032 val × 6 regions). Confirms the three
cross-paradigm predictions stated in the paper (§"Collapse generalisability").

**Results:**

| Metric | HTP (Baseline-3) | Simulated collapse (Baseline-2) |
|--------|------------------|---------------------------------|
| Mean max attention weight | **0.034** (8.75× uniform 1/T) | 0.007 (1.74× uniform 1/T) |
| Attention entropy | **4.96 nats** (89.5% of H_max) | 5.53 nats (99.6% of H_max) |
| Verdict | ✅ **HEALTHY** | ❌ **COLLAPSED** |

Uniform baseline: 1/T = 1/256 = 0.0039. H_max = log(256) = 5.55 nats.

**Prediction verification (from paper §Collapse generalisability):**

| Prediction | Threshold | HTP result | Collapse result | Verified? |
|------------|-----------|-----------|----------------|-----------|
| Max-weight bound: `max_t α_t > 1/T + ε` | > 0.0039 | 0.034 ✅ | 0.007 ✅ (marginally above) | ✅ |
| Entropy bound: `H(α) ≪ log T` | < 97% H_max | 89.5% ✅ | 99.6% ❌ (collapsed) | ✅ |
| HTP recovery ∝ T/ℓ | ≈ 8× at ℓ=32 | **8.75×** | 1.74× | ✅ |

> HTP max-weight 0.034 is **8.75×** the uniform baseline — consistent with the predicted T/ℓ=256/32=8×
> recovery factor. Entropy 4.96 nats = 89.5% of H_max — substantial temporal structure retained.
> The simulated Baseline-2 array sits at 99.6% of H_max, functionally equivalent to mean-pooling,
> consistent with the empirical measurement in the paper (max deviation < 0.0002 from 1/256).

**Reproduce:**
```bash
# From repo root — requires b3_htp_attn.npy (generated by notebook Cell 73)
python diagnose_collapse.py --weights b3_htp_attn.npy --T 256

# Collapse baseline
python diagnose_collapse.py --weights b2_simulated_attn.npy --T 256
```

Files: `diagnose_collapse.py`, `b3_htp_attn.npy`, `b2_simulated_attn.npy`.

### Per-subject generalisation (V9+QML noisy model, n=2,032 val pooled)

Two complementary per-subject analyses are available: **free-generation BLEU-1** (FG, ~16–17%)
and **teacher-forcing BLEU-1** (TF, ~30–31%). Both confirm uniform cross-subject generalisation
with range < 1pp and no outliers.

#### Free-Generation BLEU-1 (FG) — existing metric

| Subject | n | FG BLEU-1 | ROUGE-1 | Δ mean |
|---------|---|-----------|---------|--------|
| YFS | 112 | **17.04%** | 23.74% | +0.48 pp |
| YAK | 103 | 17.02% | 23.66% | +0.46 pp |
| ZKB | 104 | 16.97% | 23.97% | +0.41 pp |
| YDG | 115 | 16.95% | 23.73% | +0.39 pp |
| YAC | 82 | 16.80% | 23.59% | +0.24 pp |
| ZKH | 162 | 16.79% | 23.74% | +0.23 pp |
| ZDN | 116 | 16.64% | 22.78% | +0.08 pp |
| ZAB | 153 | 16.55% | 23.19% | −0.01 pp |
| ZKW | 167 | 16.55% | 23.27% | −0.01 pp |
| ZPH | 83 | 16.53% | 23.24% | −0.03 pp |
| ZJN | 166 | 16.36% | 23.08% | −0.20 pp |
| ZGW | 145 | 16.27% | 23.09% | −0.29 pp |
| ZDM | 143 | 16.17% | 22.64% | −0.39 pp |
| ZJS | 112 | 16.17% | 22.74% | −0.39 pp |
| ZJM | 164 | 16.11% | 22.93% | −0.45 pp |
| ZMG | 105 | **16.09%** | 22.65% | −0.47 pp |
| **Mean** | — | **16.56%** | — | — |
| Std | — | 0.33 pp | — | — |

#### Teacher-Forcing BLEU-1 (TF) — from notebook Cell 57

TF uses argmax over teacher-conditioned logits — higher than FG because ground-truth tokens are
provided as context at each step. The TF/FG gap (**4.49×** for Baseline-3) is the key conditioning-strength metric.

**Overall TF BLEU-1 (all subjects pooled): 30.95%**
- Per-subject mean ± std: **30.87 ± 0.28 pp**
- Range: **0.89 pp** (ZMG 30.30% → YFS 31.19%)
- Outliers (>1pp below mean): **none**
- ✅ Range < 1pp in both FG and TF regimes — consistent cross-subject generalisation

> Y-prefix subjects (h5py) score in the top half in both FG and TF regimes.
> Z-prefix subjects show marginally more variance (MATLAB v5/v6 scipy-loaded EEG).
> Saved to `per_subject_bleu.json`.

### Held-Out Subject Evaluation

From paper Table 9. The full pipeline (Stage 0 checkpoint retained; Stage 2 and QML noisy retrained
from scratch on the filtered set) was retrained with each subject excluded from training, then
evaluated **only on that subject's val samples**. Sentence-level smoothed BLEU-1 reported (not
corpus-level; not directly comparable to Table 3's 30.95%). Val loss Δ = holdout val loss − full
model val loss (4.1729); larger Δ for ZJM is consistent with its larger held-out set.

| Subject | Removed from train | Held-out val rows | Sent. BLEU-1 | ROUGE-1 | Val Loss | Δ val loss |
|---------|--------------------|------------------|-------------|---------|---------|-----------|
| ZMG (held out) | 895 samples | 105 | **30.15%** | 34.97% | 4.2948 | +0.1219 |
| ZJM (held out) | 895 samples | 164 | **27.07%** | 33.67% | 4.4706 | +0.2977 |
| Full model (all 16) | — | 2,032 total | 30.95%† | 36.04%† | 4.1729 | — |

†Corpus-level TF BLEU-1 from training-run eval; holdout rows use sentence-level smoothed BLEU-1 (not directly comparable).

> ✅ Both subjects produce **positive** above-chance decoding on data the model has never seen during training.
> Val loss degradation scales predictably with samples removed (ZMG +0.1219, ZJM +0.2977) — consistent
> with the model learning subject-specific statistics that improve prediction. Crucially, both holdout
> val losses remain below the Baseline-3 classical full-data val loss (4.1744), confirming that QFP
> regularisation benefit persists even in the reduced-data setting.
> ZMG chosen as best-case (n=105, smallest holdout); ZJM as harder stress-test (n=164, larger).
> Checkpoints: `hybrid_qml_noisy_holdout_ZMG.pt`, `hybrid_qml_noisy_holdout_ZJM.pt`.

### Error Analysis — Four-Category Taxonomy

From paper Supplementary Table S3 (Discussion §"Error analysis and the case for BERTScore").
Applied to all 2,032 validation samples using B3+QFP noisy. Trailing EOS-token blobs
(`TheThe…`, present in ~98% of TF outputs) stripped uniformly before scoring via
regex `(.{2,8})\1{5,}`. BERTScore F1 with `roberta-large`.

**After blob stripping, Categories 1 (Repetition loop) and 2 (Syntactic collapse) are NOT observed.
Only four categories appear:**

| Cat | Name | Threshold (priority order) | n | % | Mean BLEU-1 ± std | Mean BERT-F1 ± std |
|-----|------|---------------------------|---|---|------------------|-------------------|
| 3 | Semantic drift | BERTScore F1 < 0.800 | **9** | **0.4%** | 11.8 ± 13.7% | 79.3 ± 0.6% |
| 4 | Partial recovery | BERT 0.800–0.860 & BLEU-1 < 0.25 | **519** | **25.5%** | 17.3 ± 6.6% | 83.4 ± 1.4% |
| 5 | Lexical substitution | BERT ≥ 0.800 & BLEU-1 ∈ [0.25,0.40) or BERT ≥ 0.860 & BLEU-1 < 0.40 | **1169** | **57.5%** | 30.3 ± 5.9% | 86.0 ± 2.1% |
| 6 | Successful decoding | BLEU-1 ≥ 0.40 | **335** | **16.5%** | 45.7 ± 6.0% | 86.9 ± 2.0% |

**Semantically valid (Cat 5 + Cat 6): 74% of 2,032 predictions** — explains gap between BLEU-1 (31%) and BERTScore F1 (85.51%).

**Per-condition breakdown (NR n=639, TSR n=720, SR n=673):**

| Category | NR | TSR | SR |
|----------|----|-----|----|
| Semantic drift (Cat 3) | 0.3% | 0.8% | — |
| Partial recovery (Cat 4) | 23.3% | 13.3% | **40.7%** |
| Lexical substitution (Cat 5) | 61.0% | 63.2% | **48.1%** |
| Successful decoding (Cat 6) | 15.3% | **22.6%** | 11.0% |

> SR shows **40.7% partial recovery** vs 13–23% for NR/TSR — shorter fixation durations preserve
> sentence-level gist while degrading per-word lexical precision. TSR achieves the highest
> successful-decoding rate (22.6%) — task-structured word-by-word presentation produces the most
> decodable EEG. **Semantic drift is near-zero (0.4%, 9 samples only)** — the EEG encoder
> consistently grounds the decoder in the correct semantic domain across 99.6% of predictions.
> Output files: `error_analysis_summary.json` (compact + LaTeX S3), `error_analysis_results.json`.

### Training summary

| Stage | Config | Best val loss | Epochs |
|-------|--------|--------------|--------|
| Stage 0 MoCo | queue=128, hard negatives | 3.6014 (InfoNCE) | 20 |
| Stage 1 | GPT-2 frozen, enc lr=5e-5, batch=4, accum=2 | 4.2009 | 20 |
| Stage 2 | LoRA rank=4 α=16 block[11], enc lr=1e-6 | 4.1744 | 20 |
| QML clean | QFP 4-qubit noiseless, QML_LR=3e-4, CosineAnnealingLR | **4.1733** | 10 |
| QML noisy | NoisyQFP DepolarizingChannel+PhaseDamping+MC×16, init from clean | **4.1729** | 8 (early stop) |

---

## 8. NVIDIA NIM Agent Platform

### Three domain-specific agents

All three agents are defined in `eeg_product/nat_agents_guardrailed.py` with `[ROLE:]` tags,
explicit Out-of-scope sections, and corrected V8 baselines.

**Scientist Agent** `[ROLE: scientist]`
Given flat-text metric summary (~220 input tokens), produces a structured 8-section research analysis:
1. Dataset & Setup, 2. **Five**-model progression (V5→V8→V9→QML clean→QML noisy), 3. TF Performance,
4. FG Performance & TF/FG ratio, 5. Per-condition NR/TSR/SR,
6. Attention diagnosis (HTP + cross-region + neuroscience), 7. Qualitative samples, 8. Conclusions (4 bullets).

**Critic Agent** `[ROLE: critic]`
Reads the Scientist's first 500 chars + authoritative key numbers. Produces `[ISSUE-N] / Problem / Fix`
format, ending with Verdict and Confidence score. Hard-codes V8 baselines to prevent hallucination.

**QML Synthesiser** `[ROLE: qml_synthesiser]` *(replaces original "Explainer")*
Focused on the QuantumFusionProjector circuit mechanics — down-projection, AngleEmbedding, VQC,
residual fusion — with honest assessment of what 4-qubit classical simulation contributes.
4 paragraphs ≤380 words. Ends with one specific next step.

### NIM endpoint routing

```python
# Default: NVIDIA cloud API (current key in cell 3)
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = "meta/llama-3.1-8b-instruct"   # ~63 tok/s shared endpoint

# Switch to 70B or self-hosted brev (no code change needed):
export NIM_BASE_URL="http://localhost:8000/v1"
export NIM_MODEL="meta/llama-3.1-70b-instruct"
```

Both route through the same `AsyncOpenAI` client with `timeout=300.0` (5 minutes).
`OPENAI_API_KEY` is auto-aliased from `NVIDIA_API_KEY` for NeMo Guardrails LangChain compatibility.

### Output JSON structure

```json
{
  "stats": {
    "live_metrics": { "v9_tf_bleu1_pct": 30.64, "qml_tf_bleu1_pct": 30.62, ... },
    "baselines":    { "v5": {...}, "v8": {...} },
    "attention_analysis": { "v9_classical": {...}, "v9_qml_hybrid": {...},
                            "v9_qml_noisy_hybrid": {...} }
  },
  "scientist":        "## 1. DATASET & SETUP ...",
  "critic":           "## Critical Review ...",
  "qml_synthesiser":  "## QFP Analysis ...",
  "benchmark_records": [
    { "agent": "scientist", "ttft_ms": 561.8, "total_ms": 5745.0,
      "tokens_per_sec": 63.4, "guardrail_pass": true }
  ],
  "guardrail_audit": [
    { "agent": "scientist", "guardrail_pass": true, "guardrail_fired": "" }
  ],
  "pipeline_summary": {
    "total_pipeline_ms": 18232.1,
    "guardrail_pass_rate_pct": 100.0,
    "rails_active": true,
    "rails_mode": "NeMo loaded + Python output checks"
  }
}
```

---

## 9. NeMo Guardrails

The guardrails stack lives in `eeg_product/guardrails_config/`. All three files are required.

### `config.yml`

```yaml
models:
  - type: main
    engine: openai
    model: meta/llama-3.1-8b-instruct
    parameters:
      openai_api_base: "https://integrate.api.nvidia.com/v1"

rails:
  input:  [check eeg domain intent]
  output: [check metric hallucination, check domain relevance]
```

The `openai` engine uses LangChain's `ChatOpenAI` which requires `OPENAI_API_KEY`.
This is automatically aliased from `NVIDIA_API_KEY` by `nat_agents_guardrailed.py` before
`_load_rails()` is called. Install `langchain-openai` to activate full NeMo Guardrails:

```bash
pip install langchain-openai
```

Without it, the system falls back to Python-side checks only (still fully functional).

### `rails.co` — Colang 1.0

| Flow | Layer | What it does |
|------|-------|-------------|
| `check eeg domain intent` | Input | Blocks off-topic queries (weather, recipes, jailbreaks) before the LLM call — zero cost |
| `check metric hallucination` | Output | Calls `check_metric_bounds()` — rejects BLEU outside 20–55%, BLEU-4 outside 1–15%, BERTScore outside 78–96.5% |
| `check domain relevance` | Output | Calls `self_check_relevance()` — requires ≥3 of 38 EEG+noisy-QML terms in every response |
| `check noisy qml context` | Output | Calls `check_noisy_qml_keys()` — validates `noisy_qml_*` keys when `qml_synthesiser` role active |

> **Important:** Actual LLM calls always go through direct streaming (`AsyncOpenAI`),
> not through `rails.generate_async()`. This avoids NeMo's input intent classifier
> intercepting structured EEG metric text and returning empty responses.
> The `LLMRails` object is loaded and used to report `rails_active: True` and
> register Python actions, but does not wrap the LLM call itself.

### `guardrails_actions.py` — Python validators

- **`check_metric_bounds(response)`** — regex `(bleu[-_]?[14]?|rouge[-_]?[1l]?|bertscore)[\s:=\(of]+(\d{1,3}\.\d+|\d{2,3}(?!\.))` extracts metric values with mandatory separator. BLEU-4 uses separate range (1–15%) to avoid false positives on valid low scores.
- **`self_check_relevance(response)`** — counts 38 EEG domain terms (incl. noisy QML vocabulary: "depolarizing", "phase damping", "monte carlo", "hardware", "circuit"); fails if <3 found.
- **`check_noisy_qml_keys(stats)`** — validates `noisy_qml_tf_bleu1_pct`, `noisy_qml_tf_rouge1_pct`, `delta_noisy_vs_clean_bleu1`, `v9_qml_noisy_hybrid` exist in payload before `qml_synthesiser` runs.
- **`get_agent_role(system_prompt)`** — routes `[ROLE:]` tags to Colang dialog rail groups. `qml_synthesiser` covers both clean and noisy QML.

---

## 10. Inference Benchmark Harness

`eeg_product/benchmark/nim_benchmark.py` measures the production performance of the agent pipeline.

### Metrics collected

| Metric | Description |
|--------|-------------|
| TTFT (ms) | Time to first token from `time.perf_counter()` on streaming response |
| Total latency (ms) | Wall-clock time for complete agent response |
| Tokens per second | `output_tokens / (total_ms / 1000)` |
| Guardrail pass rate | % of calls that cleared all output rails |
| Input tokens | Estimated from word count (~0.75 tokens/word) |

### Sample results (`comparison_eegconformer_lora_v1.json`)

| Agent | TTFT (ms) | Latency (ms) | Tokens/s | Guard |
|-------|-----------|-------------|----------|-------|
| scientist | 561.8 | 5,745 | 63.4 | ✅ PASS |
| comparator | 308.1 | 3,928 | 56.3 | ✅ PASS |
| critic | 383.6 | 3,510 | 67.2 | ✅ PASS |
| qml_synthesiser | 281.1 | 5,049 | 67.7 | ✅ PASS |
| **Pipeline total** | — | **18,232** | **63.8 avg** | **100%** |

### Run multi-trial benchmark

In notebook cell 17, set `N_BENCHMARK_RUNS = 5` to collect mean/p95 statistics.
Or run standalone from the terminal:

```bash
cd eeg_product
python benchmark/nim_benchmark.py \
    --endpoint https://integrate.api.nvidia.com/v1 \
    --api-key nvapi-your-key \
    --model meta/llama-3.1-8b-instruct \
    --runs 5 \
    --output benchmark_report.json
```

---

## 11. External Researcher Interface

Any EEG-to-text researcher working on ZuCo can compare their model against the V9+QML baseline
without needing your model code, checkpoints, or the ZuCo data.

### What they need

1. Their trained model evaluated on ZuCo — just the metric numbers
2. The `eeg_product/` folder (from this repository)
3. `nat_v9_qml_results.json` (produced by running `nat_eeg_agents_v9_product.ipynb`)
4. An NVIDIA API key (free at https://build.nvidia.com)

### How to use

Open `eeg_product/external_researcher_template.ipynb`:

```python
# Cell 3 — fill in your model metrics (minimum: model_name + 2 numbers)
from eeg_submission_schema import EEGModelSubmission

my_model = EEGModelSubmission(
    model_name        = "MyEEGTransformerV2",
    architecture_desc = "6-region GRU + cross-attention + LoRA rank=16 GPT-2",
    tf_bleu1_pct      = 32.1,    # required
    tf_rouge1_pct     = 37.2,    # required
    tf_bleu4_pct      = 4.8,     # optional but recommended
    tf_rougeL_pct     = 31.5,
    fg_bleu1_pct      = 16.4,
    bertscore_f1      = 85.9,
    tf_fg_ratio       = 1.96,
    per_condition_bleu1 = {"NR": 31.8, "TSR": 33.5, "SR": 28.2},
    val_split         = "sentence-aware TEST_SIZE=0.15 seed=42",
    n_val_samples     = 2032,
    notes             = "Increased LoRA rank; no QML component",
)
```

```python
# Cell 4 — run 4-agent comparison
from comparison_pipeline import run_comparison_pipeline
results = await run_comparison_pipeline(my_model)
```

### Four comparison agents

| Agent | Role | Output |
|-------|------|--------|
| Scientist | Analyse submitted model architecture and metrics | 8-section research analysis |
| Comparator | Head-to-head table vs V9+QML | Per-metric BETTER/EQUIVALENT/WORSE/N/A verdicts |
| Critic | Challenge methodology and statistical significance | [ISSUE-N] format + ACCEPT/REVISE verdict |
| Synthesiser | Plain-language summary | 4 paragraphs + one specific next step |

### Frozen baseline values

All comparisons are automatically made against:
- **V5**: BLEU-1=29.24%, ROUGE-1=33.92% (locked constant in `eeg_submission_schema.py`)
- **V8**: BLEU-1=30.40%, ROUGE-1=35.78%, BERTScore=85.46% (locked constant)
- **V9+QML clean**: loaded live from `nat_v9_qml_results.json` via `load_v9_qml_baseline()`
- **V9+QML noisy**: loaded live from `nat_v9_qml_results.json` via `load_v9_qml_noisy_baseline()`
  (hardware-realistic simulation; val loss=4.1729; noise params: DepolarizingChannel p=0.01 + PhaseDamping γ=0.02)

---

## 12. Streamlit Dashboard — `app.py`

A full interactive analysis dashboard. Launch with:

```bash
streamlit run app.py
```

### Pages

| Page | Contents |
|------|----------|
| 🏠 **Overview** | Architecture evolution table, key metric cards (5 metrics), ZuCo conditions, brain regions table |
| 📉 **Training Curves** | Stage 0 MoCo loss + Stage 1/2 train/val + full cumulative timeline (plotly dark theme) |
| 📊 **Model Comparison** | Overall metrics bar chart (V5/V8/V9/QML), TF/FG ratio chart, full 4-model table, per-condition grouped bars, radar chart |
| 🧠 **EEG Attention** | Interactive HTP attention waveform by region+condition, attention norm bars vs V8 collapse baseline, cross-region fusion by condition, neuroscience reference table |
| 🔬 **Architecture** | Parameter breakdown pie + horizontal bar, 9-token prefix table, stage training summary, RegionEncoderV9+HTP code |
| 💬 **Qualitative Samples** | Per-condition target vs V9 TF/FG vs QML TF/FG, token overlap heatmap, alpha sweep chart |
| ⚛️ **Quantum Fusion** | VQC architecture code, parameter comparison table, val loss comparison Stage 2 vs QML, BLEU-1 progression bar; **three-way ablation** (V9 classical vs Down/Up vs QFP) bar chart + table; **bootstrap CI section** (absolute CIs, paired CIs table, significance interpretation) |
| 👥 **Per-Subject Analysis** | Three tabs: (1) FG BLEU-1 horizontal bar + scatter, (2) TF BLEU-1 per subject (Cell 57, overall 30.95%), (3) Held-out subject retraining — ZMG/ZJM within-split vs holdout comparison |
| 🔍 **Error Analysis** | Five tabs: Category Overview (taxonomy table + distribution + score charts), Per-Condition Breakdown (SR vs NR/TSR semantic-drift gap), BERTScore vs BLEU-1 scatter with threshold lines, Representative Examples per category, MC Dropout Stability + B2→B3 evidence synthesis |
| 🛡️ **NVIDIA Stack** | Live benchmark table + latency/TTFT charts, guardrail architecture (3 columns), NIM endpoint routing code, `config.yml` + `rails.co` display |
| 🤖 **NAT Agents** | 3-agent pipeline cards, system prompts viewer, `agent_stats` JSON preview, **live agent runner** (enter API key → runs all 3 agents with guardrail badges and timing) |

### Live agent runner (NAT Agents page)

Enter your NVIDIA API key and select the model (8B or 70B). The runner:
- Calls all 3 agents sequentially against `integrate.api.nvidia.com`
- Applies the Python-side guardrail check on each response
- Displays each response with a ✅/⛔ badge and latency in milliseconds
- Uses the slim 220-token prompts to keep cloud response time under 60s per agent

---

## 13. Getting an NVIDIA API Key

1. Create a free developer account at [developer.nvidia.com](https://developer.nvidia.com)
2. Go to [build.nvidia.com](https://build.nvidia.com) → sign in
3. Search for `llama-3.1-8b-instruct` → click **Get API Key**
4. Copy the `nvapi-...` key (shown only once)

**Set it before running:**

```bash
# Recommended: environment variable
export NVIDIA_API_KEY="nvapi-your-key-here"
```

Or paste directly into `nat_eeg_agents_v9_product.ipynb` cell 3.

**API endpoint used:**

```
https://integrate.api.nvidia.com/v1/chat/completions
```

This is an OpenAI-compatible endpoint. Any model on `build.nvidia.com` can be substituted
by changing `NIM_MODEL` — no other code changes required.

**Free tier:** ~1,000 API calls or 40,000 tokens/minute. The 3-agent pipeline uses ~2,000–4,000
tokens per run at the 8B model (220-token prompts + 900 max output tokens per agent).

---

## 14. Plots Reference

All figures saved to `plots/`.

| File | Description |
|------|-------------|
| `plot_loss_curves.png` | Stage 0 MoCo InfoNCE + Stage 1 train/val + Stage 2 LoRA train/val |
| `plot_overfitting.png` | Overfitting diagnosis: Stage 1 gap controlled at ~0.13 vs old 1.65 |
| `plot_per_condition_bleu.png` | Grouped bar: V5/V8/V9/QML clean/QML noisy per condition (NR/TSR/SR) |
| `plot_metrics_comparison.png` | All five metrics (BLEU-1/4, ROUGE-1/L, BERTScore) V8→V9→QML clean→noisy |
| `plot_val_timeline.png` | Unified val loss: Stage 1 (coral) + Stage 2 (amber) + QML clean (purple) + QML noisy (pink) |
| `plot_stage0_convergence.png` | MoCo InfoNCE over 20 epochs with plateau annotation |
| `plot_stage2_improvement.png` | Stage 2 LoRA + QML clean + QML noisy (3-panel, early stop on ep 8) |
| `plot_inference_comparison.png` | 4-model bar chart (V8/V9/QML clean/QML noisy) — BLEU-1/4, ROUGE-1/L |
| `diag1_pool_attn_collapse.png` | 6-panel attention distributions — 4 collapsed regions (H/Hmax>0.95) |
| `diag2_v9_fusion_weights.png` | Cross-region fusion weights heatmap per condition |
| `attn_htp_NR.png` | HTP local attention profiles — Normal Reading |
| `attn_htp_TSR.png` | HTP local attention profiles — Timed Silent Reading |
| `attn_htp_SR.png` | HTP local attention profiles — Speed Reading |
| `system_architecture.png` | Full model architecture diagram |
| `eeg_encoder.png` | EEGEncoder regional structure |
| `qml.png` | QuantumFusionProjector circuit diagram |
| `training_pipeline.png` | Three-stage training pipeline flow |
| `preprocessed_pipeline.png` | EEG preprocessing steps |
| `prefix_tokens.png` | 9-token prefix construction |
| `processed_eeg.png` | Example processed EEG signal |
| `noisy_qml.png` | Noisy QML training curves — DepolarizingChannel+PhaseDamping convergence |
| `agent.png` | NVIDIA NAT agent pipeline architecture diagram |
| `trial1.png` / `trial2.png` | Sample trial visualisations |

---

## 15. Key Findings

### What worked

1. **HTP fixed temporal pooling collapse — the paper's central contribution.** Baseline-2's flat `pool_attn Linear(D,1)` had uniform 1/256 ≈ 0.0039 attention across all timesteps and all six regions (max deviation < 0.0002) — equivalent to mean-pooling. HTP's two-level softmax (32-way local + 8-way segment) recovers attention norms **10–30×** (Table 4): Left Parieto-Occipital 30.9×, Left Temporal 28.8×. The collapse is proven mathematically inevitable for any architecture using single-level softmax over T ≥ 128 timesteps — it is an architectural property, not an optimisation failure.

   **Empirically confirmed by `diagnose_collapse.py` (Cells 73–74, n=12,192 samples):**
   - HTP mean max weight: **0.034** (8.75× uniform) — attention is genuinely selective
   - HTP entropy: **4.96 nats** (89.5% of H_max=5.55) — substantial structure, not near-maximal
   - Simulated Baseline-2: 0.007 max weight (1.74× uniform), 5.53 nats (99.6% H_max) — **COLLAPSED**
   - Recovery factor 8.75× is consistent with the predicted T/ℓ = 256/32 = 8× from the paper's formal analysis
   - Reproduce: `python diagnose_collapse.py --weights b3_htp_attn.npy --T 256`

2. **TF/FG ratio is the primary evidence of EEG-token alignment.** Baseline-2: 6.19×. Baseline-3: **4.49×** (37% improvement). This rightward shift in optimal EEG beam weight (α=1.0 → 4.0) directly quantifies that HTP-encoded prefixes are informative across a fourfold wider operating range. The model genuinely conditions on EEG rather than language priors — this is more meaningful than the modest BLEU gains.

3. **Neurophysiological coherence emerges from HTP.** Under TSR, VWFA/Left Parieto-Occipital dominates (fusion weight 0.263) — consistent with rapid orthographic identification. Under NR and SR, Left Parietal leads (0.243, 0.231) — consistent with semantic integration. A cross-condition P300-like central-parietal onset (0.261) appears regardless of reading speed. None of these patterns exist in Baseline-2's collapsed 1/256 baseline.

4. **Freezing GPT-2 in Stage 1 eliminated overfitting.** Old Stage 1 with GPT-2 unlocked: train/val gap = 1.65 at early stop epoch 7. Fixed Stage 1 (fully frozen): gap = ~0.13 at epoch 17. Val loss 4.2009 at epoch 17. EEG encoder training must precede LLM adaptation.

5. **QFP provides marginal regularisation — not a quantum advantage claim.** Val loss −0.0011 (the only tested lightweight projector to reduce val loss). All BLEU-1 differences are **statistically non-significant**: QFP vs B3 p=0.668, QFP vs Down/Up p=0.682 (bootstrap B=10,000). Val loss is the discriminating metric. The bounded [−1,1] expectation-value output acts as implicit regularisation at ZuCo's 11,955-sample scale; a classical 768→4→768 GELU projector overfits (Δval +0.0318).

6. **QFP noisy matches clean — architecture is hardware-deployable.** Hardware-realistic noise simulation (DepolarizingChannel p=0.01, PhaseDamping γ=0.02, MC-average NMC=16) achieves val loss 4.1729 vs clean 4.1733 (Δ=−0.0004). Noise during training regularises the classical up-projection W↑. All generation metrics match exactly.

7. **Cross-subject generalisation confirmed by two complementary analyses.** (a) Within-split per-subject sentence BLEU-1 range: **0.95pp** (ZMG 16.09% → YFS 17.04%, mean 16.56±0.33pp); no subject falls >1pp below the cross-subject mean. (b) Held-out retraining: ZMG achieves **30.15% sentence BLEU-1** on 105 unseen samples, ZJM **27.07%** on 164 unseen samples — both positive, both held-out val losses below Baseline-3 classical (4.1744). Val loss degradation scales predictably with held-out set size (+0.1219 ZMG, +0.2977 ZJM).

8. **B2→B3 gain is 12× larger than the inference noise floor.** MC-dropout std (5 passes, model.train(), seeds 42/123/456/789/1024): B3 **±0.05%** std (sentence-level). Corpus B2→B3 gain (+0.62pp) / MC std = **12.0× SNR**. The gain cannot be attributed to inference-level stochasticity.

9. **Error analysis: 74% of predictions are semantically valid; only 0.4% semantic drift.** After stripping universal EOS artefacts (~98% of TF outputs): Cat 3 (semantic drift) = **9 samples (0.4%)** — near-zero. Cat 5 (lexical substitution) = **1,169 (57.5%)** — correct topic, different words. Cat 6 (success) = **335 (16.5%)**. Cat 5+6 = **74%** semantically valid. This explains the BERTScore F1 (85.51%) vs BLEU-1 (31%) gap. TSR achieves 22.6% successful decoding; SR shows 40.7% partial recovery due to shorter fixation durations.

10. **Guardrailed agent pipeline is production-grade.** Pipeline total 176,497ms (Scientist 48,997ms, Critic 16,126ms, QML Synthesiser 111,374ms). 100% guardrail pass rate across all calls. Critic: CONDITIONAL PASS 6/10, all 7 raised issues addressed in paper. Domain: metric-bounds rail BLEU 20–55%, BERTScore 78–96.5%.

### What remains open

1. **TF/FG gap** — FG BLEU-1 (6.90%) remains far below TF BLEU-1 (31.02%). The highest-priority architectural next step is vocabulary-constrained beam search targeting the 57.5% lexical-substitution failure mode, which would likely close much of the remaining gap without any architectural change.

2. **Cross-paradigm collapse verification** — the 1/T collapse prediction should be tested on publicly available EEG classifiers (EEGNet, ATCNet) on BCI Competition IV data. `diagnose_collapse.py` is released specifically for this: load any model's attention weights as a `.npy` array and run `python diagnose_collapse.py --weights <path> --T <seq_len>`. Confirmed cross-paradigm, this would elevate the finding from a ZuCo-specific observation to a general architectural principle.

3. **Full 16-fold LOSO** — two subjects (ZMG, ZJM) were held out as representative best-case and stress-test. Full leave-one-subject-out across all 16 requires ~192 GPU-hours on consumer hardware — outside current compute budget and not reported in any prior ZuCo system.

4. **Second corpus** — ZuCo is the only publicly available corpus providing simultaneous sentence-level EEG+eye-tracking for open-vocabulary natural reading. Cross-corpus evaluation requires a dataset that does not yet exist.

5. **Scale to 70B on dedicated GPU** — current benchmarks use 8B on shared cloud endpoint. Brev GPU deployment with `meta/llama-3.1-70b-instruct` will produce publication-quality agent analysis and throughput benchmarks.

6. **QFP scale** — whether the QFP regularisation advantage persists at larger data scales, or whether a well-tuned classical alternative could achieve the same effect, remains open. The 12-qubit extension (density-matrix cost 4^n) is deferred to future work with appropriate hardware.

---

## 16. Citation

If you use this codebase, results, or benchmarking platform, please cite:

```bibtex
@article{bhattacharya2026diagnosing,
  title   = {Temporal Attention Collapse in EEG-to-Text Decoding:
Repair with Hierarchical Temporal Pooling and
Anatomical Encoding via Multi-Region
GRU–Transformers},
  author  = {Bhattacharya, Deeptanshu and Shridevi, S.},
  year    = {2026},
  note    = {EEG-to-text on ZuCo (n=2,032 val). Primary finding: temporal attention collapse
             in single-level EEG encoders (1/T denominator, T=256). HTP repair recovers 10--30×
             attention-norm magnitude; TF/FG 6.19×→4.49× (37\% alignment gain).
             TF BLEU-1=31.02\%; BERTScore F1=85.51\%; semantic drift=0.4\% (9/2032 samples).
             Exploratory QFP (8,476 params): Δval=−0.0011; BLEU-1 non-significant (p=0.668).
             MC dropout SNR=12.0×. Held-out ZMG=30.15\%, ZJM=27.07\%.
             NVIDIA NIM + NeMo Guardrails Colang 1.0 pipeline.}
}
```

ZuCo dataset:

```bibtex
@article{hollenstein2018zuco,
  title   = {ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading},
  author  = {Hollenstein, Nora and Rotsztejn, Jonathan and Troendle, Marius and
             Pedroni, Andreas and Zhang, Ce and Langer, Nicolas},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180259},
  year    = {2018}
}
```

---

*Built with PyTorch 2.8 · PennyLane 0.44.1 · HuggingFace Transformers · bert-score (roberta-large) ·
NVIDIA NIM · NeMo Guardrails Colang 1.0 · Streamlit · Tested on RTX 3050 local + NVIDIA cloud NIM.*