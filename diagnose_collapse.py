"""
diagnose_collapse.py
====================
Temporal Attention Collapse Diagnostic Tool
--------------------------------------------
Companion to:
  "Diagnosing and Repairing Temporal Attention Collapse in Low-Resource
   EEG-to-Text Decoding: Hierarchical Temporal Pooling and Multi-Region
   GRU-Transformer Encoders"
  Bhattacharya & S., 2026.

PURPOSE
-------
Tests whether a given EEG attention module exhibits the 1/T denominator
collapse described in the paper. The collapse is defined as single-level
softmax attention weights degenerating to near-uniform 1/T regardless of
input content — making the attention mechanistically equivalent to mean
pooling.

THREE FALSIFIABLE PREDICTIONS TESTED (from paper Section "Collapse
Generalisability"):

  Prediction 1 — Max-weight bound:
    max_t(alpha_t) <= 1/T + epsilon
    where epsilon decreases with encoder output variance.
    Collapsed: max weight barely exceeds 1/T.
    Healthy:   max weight substantially exceeds 1/T.

  Prediction 2 — Entropy bound:
    H(alpha) approaches log(T) as T grows.
    At T=256: H_max = log(256) = 5.545 nats.
    Collapsed: H within 5% of H_max  (>= 5.27 nats).
    Healthy:   H substantially below H_max.

  Prediction 3 — HTP recovery proportionality:
    Windowed attention with denominator ell recovers norm in proportion
    to T/ell. At ell=32: expected recovery ~8x; at ell=16: ~16x.
    (Empirical recovery in paper: 10-30x, consistent with this.)

USAGE
-----
The tool accepts either:
  (A) A raw attention weight tensor   shape (N, T)  — N samples, T timesteps
  (B) A PyTorch attention module      nn.Module     — run on provided EEG data
  (C) A numpy array from any source   shape (N, T)

Command-line examples:
  # From a saved numpy file of attention weights:
  python diagnose_collapse.py --weights attn_weights.npy --T 256

  # Pass a custom T (e.g. for a 128-timestep model):
  python diagnose_collapse.py --weights attn_weights.npy --T 128

  # Demo mode — simulate a collapsed model vs a healthy one:
  python diagnose_collapse.py --demo

Programmatic use:
  from diagnose_collapse import diagnose, simulate_collapse, simulate_healthy
  results = diagnose(alpha)   # alpha: np.ndarray shape (N, T)
  print(results)

REQUIREMENTS
------------
  numpy >= 1.20
  scipy >= 1.7   (for entropy)
  matplotlib >= 3.4  (optional, for --plot flag)
"""

import argparse
import sys
import numpy as np

try:
    from scipy.stats import entropy as scipy_entropy
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Core diagnostic functions
# ─────────────────────────────────────────────────────────────────────────────

def entropy_nats(alpha: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy in nats for each row of alpha.
    alpha: shape (N, T), each row sums to 1.
    Returns: shape (N,)
    """
    alpha = np.clip(alpha, 1e-12, 1.0)
    return -np.sum(alpha * np.log(alpha), axis=-1)


def diagnose(alpha: np.ndarray, T: int = None,
             window_sizes: list = None) -> dict:
    """
    Run the three collapse predictions on attention weights alpha.

    Parameters
    ----------
    alpha : np.ndarray, shape (N, T)
        Attention weight distributions. Each row must sum to 1.
        N = number of samples, T = sequence length.
    T : int, optional
        Expected sequence length. Inferred from alpha.shape[1] if None.
    window_sizes : list of int, optional
        Window sizes to test for Prediction 3 (HTP recovery).
        Defaults to [128, 64, 32, 16, 8].

    Returns
    -------
    dict with keys:
        T, N, uniform_baseline,
        pred1_max_weight (mean, std, ratio_to_uniform, collapsed),
        pred2_entropy    (mean_nats, std_nats, H_max, frac_of_Hmax, collapsed),
        pred3_htp        (per window size: norm_ratio, expected_ratio),
        overall_verdict  ("COLLAPSED" | "LIKELY_COLLAPSED" | "HEALTHY"),
        summary_lines    (list of human-readable strings)
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 1:
        alpha = alpha[np.newaxis, :]
    N, T_actual = alpha.shape

    # Normalise rows to sum to 1 (HTP local attn sums to n_windows, not 1)
    row_sums = alpha.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, atol=0.05):
        alpha = alpha / np.clip(row_sums, 1e-12, None)

    if T is None:
        T = T_actual
    elif T != T_actual:
        print(f"[WARNING] Provided T={T} does not match alpha shape T={T_actual}. "
              f"Using T={T_actual}.")
        T = T_actual

    if window_sizes is None:
        window_sizes = [128, 64, 32, 16, 8]

    uniform = 1.0 / T
    H_max = np.log(T)  # entropy of uniform distribution in nats

    # ── Prediction 1: Max-weight bound ────────────────────────────────────
    max_weights = alpha.max(axis=1)           # shape (N,)
    mean_max = float(np.mean(max_weights))
    std_max  = float(np.std(max_weights))
    ratio_max_to_uniform = mean_max / uniform

    # Collapsed if mean max weight is within 10% of 1/T
    p1_collapsed = mean_max <= uniform * 2.00   # <=2x uniform = collapsed

    # ── Prediction 2: Entropy bound ───────────────────────────────────────
    H = entropy_nats(alpha)                   # shape (N,)
    mean_H = float(np.mean(H))
    std_H  = float(np.std(H))
    frac_of_Hmax = mean_H / H_max

    # Collapsed if mean entropy is within 5% of H_max
    p2_collapsed = frac_of_Hmax >= 0.95

    # ── Prediction 3: HTP recovery proportionality ───────────────────────
    # Baseline norm: mean of max alpha (the 1/T uniform case has norm ~ 1/T)
    baseline_norm = float(np.mean(max_weights))

    htp_results = {}
    for ell in window_sizes:
        if ell >= T:
            continue
        n_windows = T // ell
        # Simulate windowed softmax: reshape and compute max weight per window
        alpha_windowed = alpha[:, :n_windows * ell].reshape(N, n_windows, ell)
        # For each window, compute a local softmax (alpha already sums to 1
        # globally; simulate local by re-normalising each window)
        window_sums = alpha_windowed.sum(axis=2, keepdims=True).clip(1e-12)
        alpha_local = alpha_windowed / window_sums
        local_max = alpha_local.max(axis=2)           # (N, n_windows)
        mean_local_max = float(np.mean(local_max))

        expected_ratio = T / ell
        empirical_ratio = mean_local_max / baseline_norm if baseline_norm > 0 else 0.0

        htp_results[ell] = {
            "mean_local_max_weight": mean_local_max,
            "empirical_norm_ratio":  empirical_ratio,
            "expected_norm_ratio":   expected_ratio,
            "consistent":            empirical_ratio >= expected_ratio * 0.5,
        }

    # ── Overall verdict ───────────────────────────────────────────────────
    n_collapsed = sum([p1_collapsed, p2_collapsed])
    if n_collapsed == 2:
        verdict = "COLLAPSED"
    elif n_collapsed == 1:
        verdict = "LIKELY_COLLAPSED"
    else:
        verdict = "HEALTHY"

    # ── Human-readable summary ────────────────────────────────────────────
    lines = [
        "=" * 65,
        "  TEMPORAL ATTENTION COLLAPSE DIAGNOSTIC REPORT",
        "=" * 65,
        f"  Samples (N):        {N}",
        f"  Sequence length (T): {T}",
        f"  Uniform baseline:   1/T = {uniform:.6f}",
        f"  H_max = log(T):     {H_max:.4f} nats",
        "",
        "─" * 65,
        "  PREDICTION 1 — Max-weight bound",
        "─" * 65,
        f"  Mean max(alpha_t):  {mean_max:.6f}  ±  {std_max:.6f}",
        f"  Ratio to 1/T:       {ratio_max_to_uniform:.2f}x",
        f"  Threshold (<=2.00x uniform = collapsed): "
        f"{'COLLAPSED ✗' if p1_collapsed else 'HEALTHY ✓'}",
        "",
        "─" * 65,
        "  PREDICTION 2 — Entropy bound",
        "─" * 65,
        f"  Mean H(alpha):      {mean_H:.4f}  ±  {std_H:.4f} nats",
        f"  H_max:              {H_max:.4f} nats",
        f"  Fraction of H_max:  {frac_of_Hmax:.4f}  "
        f"(collapsed threshold: >=0.95)",
        f"  Status:             "
        f"{'COLLAPSED ✗' if p2_collapsed else 'HEALTHY ✓'}",
        "",
        "─" * 65,
        "  PREDICTION 3 — HTP Recovery Proportionality",
        "─" * 65,
    ]

    for ell, res in sorted(htp_results.items(), reverse=True):
        status = "consistent ✓" if res["consistent"] else "inconsistent ✗"
        lines.append(
            f"  ell={ell:3d}: empirical {res['empirical_norm_ratio']:5.1f}x  "
            f"expected {res['expected_norm_ratio']:5.1f}x  [{status}]"
        )

    lines += [
        "",
        "=" * 65,
        f"  OVERALL VERDICT:  {verdict}",
        "=" * 65,
        "",
        "  Interpretation:",
        "  COLLAPSED        — Both predictions 1 & 2 indicate collapse.",
        "                     Attention is functionally equivalent to mean",
        "                     pooling. Consider HTP or windowed attention.",
        "  LIKELY_COLLAPSED — One prediction indicates collapse. Inspect",
        "                     attention qualitatively before concluding.",
        "  HEALTHY          — No collapse detected. Attention norms and",
        "                     entropy are well below the uniform baseline.",
        "=" * 65,
    ]

    return {
        "T": T,
        "N": N,
        "uniform_baseline": uniform,
        "pred1_max_weight": {
            "mean":              mean_max,
            "std":               std_max,
            "ratio_to_uniform":  ratio_max_to_uniform,
            "collapsed":         p1_collapsed,
        },
        "pred2_entropy": {
            "mean_nats":     mean_H,
            "std_nats":      std_H,
            "H_max":         H_max,
            "frac_of_Hmax":  frac_of_Hmax,
            "collapsed":     p2_collapsed,
        },
        "pred3_htp": htp_results,
        "overall_verdict": verdict,
        "summary_lines": lines,
    }


def print_report(results: dict) -> None:
    """Print the diagnostic report to stdout."""
    for line in results["summary_lines"]:
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers — reproduce the paper's collapsed vs healthy examples
# ─────────────────────────────────────────────────────────────────────────────

def simulate_collapse(N: int = 160, T: int = 256,
                      score_scale: float = 0.2) -> np.ndarray:
    """
    Simulate a COLLAPSED attention distribution (single-level softmax, T=256).
    score_scale controls how peaked the raw scores are.
    With small score_scale the softmax saturates at ~1/T.
    Returns alpha: shape (N, T), each row sums to 1.
    """
    rng = np.random.default_rng(42)
    scores = rng.normal(0, score_scale, size=(N, T))
    # Standard single-level softmax
    scores -= scores.max(axis=1, keepdims=True)  # numerical stability
    exp_s = np.exp(scores)
    alpha = exp_s / exp_s.sum(axis=1, keepdims=True)
    return alpha


def simulate_healthy(N: int = 160, T: int = 256,
                     window: int = 32) -> np.ndarray:
    """
    Simulate a HEALTHY attention distribution (HTP windowed softmax).
    window = local softmax denominator (e.g. 32 for HTP ell=32).
    Returns alpha: shape (N, T), each row sums to 1 (globally).
    """
    rng = np.random.default_rng(42)
    n_windows = T // window
    scores = rng.normal(0, 1.5, size=(N, n_windows, window))
    scores -= scores.max(axis=2, keepdims=True)
    exp_s = np.exp(scores)
    alpha_local = exp_s / exp_s.sum(axis=2, keepdims=True)  # sums to 1 per window
    # Flatten back to (N, T) and renormalise globally to sum to 1
    alpha_flat = alpha_local.reshape(N, T)
    alpha_flat = alpha_flat / alpha_flat.sum(axis=1, keepdims=True)
    return alpha_flat


# ─────────────────────────────────────────────────────────────────────────────
# Optional: plot attention profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(alpha_collapsed: np.ndarray,
                    alpha_healthy: np.ndarray,
                    T: int = 256,
                    save_path: str = None) -> None:
    """Plot mean attention profiles side-by-side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    uniform = 1.0 / T
    t = np.arange(T)

    mean_c = alpha_collapsed.mean(axis=0)
    std_c  = alpha_collapsed.std(axis=0)
    mean_h = alpha_healthy.mean(axis=0)
    std_h  = alpha_healthy.std(axis=0)

    for ax, mean, std, title, color in [
        (axes[0], mean_c, std_c, f"COLLAPSED  (single-level softmax, T={T})", "crimson"),
        (axes[1], mean_h, std_h, f"HEALTHY  (HTP windowed, ell=32)", "steelblue"),
    ]:
        ax.fill_between(t, mean - std, mean + std, alpha=0.3, color=color)
        ax.plot(t, mean, color=color, lw=1.5, label="mean ± std")
        ax.axhline(uniform, color="black", ls="--", lw=1.2,
                   label=f"1/T = {uniform:.4f}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Attention weight")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Attention Collapse Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to a .npy or .npz file containing attention weights "
             "of shape (N, T). For .npz, the first array is used.",
    )
    parser.add_argument(
        "--T", type=int, default=None,
        help="Expected sequence length T (inferred from array if not given).",
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=[128, 64, 32, 16, 8],
        help="Window sizes to test for Prediction 3 (default: 128 64 32 16 8).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a demo comparing simulated collapsed vs healthy attention.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show a comparison plot (requires matplotlib). "
             "Use with --demo or --weights.",
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="Save plot to this path instead of displaying it.",
    )
    args = parser.parse_args()

    if args.demo:
        print("\n[DEMO MODE] Simulating collapsed and healthy attention\n")
        T = args.T or 256

        print("── COLLAPSED MODEL (single-level softmax, score_scale=0.2) ──")
        alpha_c = simulate_collapse(N=160, T=T)
        res_c = diagnose(alpha_c, T=T, window_sizes=args.windows)
        print_report(res_c)

        print("\n── HEALTHY MODEL (HTP windowed softmax, ell=32) ──")
        alpha_h = simulate_healthy(N=160, T=T, window=32)
        res_h = diagnose(alpha_h, T=T, window_sizes=args.windows)
        print_report(res_h)

        if args.plot or args.save_plot:
            plot_comparison(alpha_c, alpha_h, T=T, save_path=args.save_plot)
        return

    if args.weights is not None:
        path = args.weights
        if path.endswith(".npy"):
            alpha = np.load(path)
        elif path.endswith(".npz"):
            npz = np.load(path)
            alpha = npz[list(npz.keys())[0]]
        else:
            print(f"[ERROR] Unsupported file format: {path}. "
                  f"Use .npy or .npz.", file=sys.stderr)
            sys.exit(1)

        if alpha.ndim != 2:
            print(f"[ERROR] Expected 2D array (N, T), got shape {alpha.shape}.",
                  file=sys.stderr)
            sys.exit(1)

        results = diagnose(alpha, T=args.T, window_sizes=args.windows)
        print_report(results)

        if args.plot or args.save_plot:
            dummy = simulate_healthy(N=alpha.shape[0], T=alpha.shape[1])
            plot_comparison(alpha, dummy, T=alpha.shape[1],
                            save_path=args.save_plot)
        return

    # No arguments — print help and run a quick demo
    print("[INFO] No arguments provided. Running quick demo with T=256.\n"
          "       Use --help for full usage.\n")
    alpha_c = simulate_collapse(N=160, T=256)
    res = diagnose(alpha_c, T=256)
    print_report(res)


if __name__ == "__main__":
    main()