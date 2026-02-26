"""
Model comparison: ACT-R (Pavlik & Anderson, 2005) vs FE-ACT-R
Powell optimizer with random restarts, parallelized across subjects.

USAGE:
    python fit_models.py

Edit DATA_PATH, OUTPUT_DIR, N_RESTARTS, and N_WORKERS below.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import json
import time
import os
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH  = 'newdata.csv'  # path to your CSV data file
OUTPUT_DIR = '.'                            # directory for output files
N_RESTARTS = 15                             # random restarts per subject/model
N_WORKERS  = None                           # None = use all available CPU cores

# ─────────────────────────────────────────────
# FIXED PARAMETERS
# ─────────────────────────────────────────────
TAU  = -0.8
S    = 0.25
BETA = np.sqrt(3) / (np.pi * S)   # ≈ 2.21, fixed

# ─────────────────────────────────────────────
# PARAMETER BOUNDS
# ─────────────────────────────────────────────
ACTR_BOUNDS = [(0, 1), (0, 1), (0.3, 1), (0.5, 2)]   # c, phi, t0, F
FEAR_BOUNDS = [(0, 1), (0, 10), (0.3, 1), (0.5, 2)]   # d, w1, t0, F


# ─────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def compute_activation_actr(enc_times, t_query, c, phi):
    n = len(enc_times)
    if n == 0:
        return np.nan
    decay_rates = np.empty(n)
    decay_rates[0] = phi
    for i in range(1, n):
        dt = np.maximum(enc_times[i] - enc_times[:i], 1e-10)
        A  = np.log(max(np.sum(dt ** (-decay_rates[:i])), 1e-10))
        decay_rates[i] = c * np.exp(A) + phi
    dt_q = np.maximum(t_query - enc_times, 1e-10)
    return np.log(max(np.sum(dt_q ** (-decay_rates)), 1e-10))

def compute_activation_fear(enc_times, t_query, d, w1):
    n = len(enc_times)
    if n == 0:
        return np.nan
    weights = np.empty(n)
    weights[0] = w1
    for i in range(1, n):
        dt = np.maximum(enc_times[i] - enc_times[:i], 1e-10)
        A  = np.log(max(np.sum(weights[:i] * (dt ** (-d))), 1e-10))
        p  = float(sigmoid((A - TAU) / S))
        p  = np.clip(p, 1e-10, 1 - 1e-10)
        weights[i] = -np.log(p)
    dt_q = np.maximum(t_query - enc_times, 1e-10)
    return np.log(max(np.sum(weights * (dt_q ** (-d))), 1e-10))


# ─────────────────────────────────────────────
# LOG-LIKELIHOOD COMPONENTS
# ─────────────────────────────────────────────

def log_lik_accuracy(A, is_correct):
    x = (A - TAU) / S
    return -np.log1p(np.exp(-x)) if is_correct else -np.log1p(np.exp(x))

def log_lik_rt(rt_shifted, A, F):
    alpha = F * np.exp(-A)
    if alpha <= 0 or rt_shifted <= 0:
        return -1e6
    z = rt_shifted / alpha
    return (np.log(BETA) - np.log(alpha)
            + (BETA - 1) * np.log(z)
            - 2 * np.log1p(z ** BETA))


# ─────────────────────────────────────────────
# SEQUENCE PRE-PROCESSING
# ─────────────────────────────────────────────

def build_fact_sequences(fact_df):
    """Convert a fact's trial rows into a list of (enc_snapshot, t_query, is_correct, rt_s)."""
    rows = fact_df.sort_values('trial').reset_index(drop=True)
    enc_list = []
    queries  = []
    for _, row in rows.iterrows():
        t = row['time']
        if row['type'] == 'study':
            enc_list.append(t)
        else:
            queries.append((np.array(enc_list, dtype=float), float(t),
                            bool(row['isCorrect']), float(row['RT_s'])))
            enc_list.append(t)
    return queries

def nll_fact_actr(queries, c, phi, t0, F):
    total = 0.0
    for enc_snap, tq, correct, rt_s in queries:
        A = compute_activation_actr(enc_snap, tq, c, phi)
        if not np.isfinite(A):
            total += 1e6; continue
        total -= log_lik_accuracy(A, correct) + log_lik_rt(rt_s - t0, A, F)
    return total

def nll_fact_fear(queries, d, w1, t0, F):
    total = 0.0
    for enc_snap, tq, correct, rt_s in queries:
        A = compute_activation_fear(enc_snap, tq, d, w1)
        if not np.isfinite(A):
            total += 1e6; continue
        total -= log_lik_accuracy(A, correct) + log_lik_rt(rt_s - t0, A, F)
    return total


# ─────────────────────────────────────────────
# TOP-LEVEL OBJECTIVE FUNCTIONS
# (must be top-level to be picklable for multiprocessing)
# ─────────────────────────────────────────────

def _actr_objective(args):
    params, seqs, facts, bounds = args
    lowers = np.array([b[0] for b in bounds])
    uppers = np.array([b[1] for b in bounds])
    params = np.clip(params, lowers, uppers)
    c, phi, t0, F = params[0], params[1], params[2], params[3]
    total = sum(nll_fact_actr(seqs[fact], c, phi, t0, F) for fact in facts)
    return total if np.isfinite(total) else 1e9

def _fear_objective(args):
    params, seqs, facts, bounds = args
    lowers = np.array([b[0] for b in bounds])
    uppers = np.array([b[1] for b in bounds])
    params = np.clip(params, lowers, uppers)
    d, w1, t0, F = params[0], params[1], params[2], params[3]
    total = sum(nll_fact_fear(seqs[fact], d, w1, t0, F) for fact in facts)
    return total if np.isfinite(total) else 1e9


# ─────────────────────────────────────────────
# PER-PARTICIPANT FITTING
# ─────────────────────────────────────────────

def _powell(objective, x0, bounds, coarse=False):
    opts = ({'maxiter': 10000, 'maxfev': 800,  'ftol': 1e-3, 'xtol': 1e-3} if coarse else
            {'maxiter': 10000, 'maxfev': 10000, 'ftol': 1e-6, 'xtol': 1e-6})
    return minimize(objective, x0, method='Powell', bounds=bounds, options=opts)

def fit_participant(subj_data, facts, model, n_restarts=N_RESTARTS):
    """Fit one model ('actr' or 'fear') to one participant's data."""
    assert model in ('actr', 'fear')
    # Set t0 upper bound to 95% of subject's fastest RT to avoid rt_shifted <= 0
    min_rt = subj_data[subj_data['type'] == 'test']['RT_s'].min()
    t0_upper = min(0.95 * min_rt, 1.0)   # never exceed global upper bound of 1.0
    t0_lower = min(0.3, t0_upper * 0.5)  # ensure lower < upper
    bounds = list(ACTR_BOUNDS if model == 'actr' else FEAR_BOUNDS)
    bounds[2] = (t0_lower, t0_upper)     # t0 is always index 2: [c/d, phi/w1, t0, F]
    lowers = np.array([b[0] for b in bounds])
    uppers = np.array([b[1] for b in bounds])

    seqs = {fact: build_fact_sequences(subj_data[subj_data['item'] == fact])
            for fact in facts}

    obj_fn = _actr_objective if model == 'actr' else _fear_objective
    obj    = lambda p: obj_fn((p, seqs, facts, bounds))

    rng  = np.random.default_rng(42)
    best = None
    for _ in range(n_restarts):
        x0  = lowers + rng.random(len(bounds)) * (uppers - lowers)
        res = _powell(obj, x0, bounds, coarse=True)
        if best is None or res.fun < best.fun:
            best = res
    # Final polish from best start
    best = _powell(obj, best.x, bounds, coarse=False)
    return best


# ─────────────────────────────────────────────
# WORKER FUNCTION (top-level for pickling)
# ─────────────────────────────────────────────

def _fit_subject(args):
    """Fit both models for one subject. Called by process pool."""
    subj, subj_data, n_restarts = args
    facts  = sorted(subj_data['item'].unique())
    n_test = len(subj_data[subj_data['type'] == 'test'])

    res_a = fit_participant(subj_data, facts, 'actr', n_restarts)
    res_f = fit_participant(subj_data, facts, 'fear', n_restarts)

    actr = {
        'nll': float(res_a.fun), 'c': float(res_a.x[0]),
        'phi': float(res_a.x[1]), 't0': float(res_a.x[2]), 'F': float(res_a.x[3]),
        'n_params': 4, 'converged': bool(res_a.success)
    }
    fear = {
        'nll': float(res_f.fun), 'd': float(res_f.x[0]),
        'w1': float(res_f.x[1]), 't0': float(res_f.x[2]), 'F': float(res_f.x[3]),
        'n_params': 4, 'converged': bool(res_f.success)
    }
    return subj, n_test, facts, actr, fear


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("Loading and filtering data...")
    df = pd.read_csv(DATA_PATH)
    df['RT_s'] = df['RT'] / 1000.0
    test_mask = df['type'] == 'test'
    rt_mask   = (df['RT_s'] >= 0.2) & (df['RT_s'] <= 10.0)
    n_before  = len(df)
    df        = df[~test_mask | rt_mask].copy()
    print(f"Trials: {n_before} → {len(df)} ({n_before - len(df)} outliers removed)")

    subjects  = sorted(df['subj'].unique())
    n_workers = N_WORKERS or cpu_count()
    print(f"Subjects: {len(subjects)} | Workers: {n_workers} | Restarts: {N_RESTARTS}")

    # Build per-subject args (DataFrames are picklable)
    job_args = [(subj, df[df['subj'] == subj].copy(), N_RESTARTS)
                for subj in subjects]

    t_start = time.time()
    print(f"\nFitting all subjects in parallel ({n_workers} cores)...")

    with Pool(processes=n_workers) as pool:
        results = pool.map(_fit_subject, job_args)

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.0f}s ({elapsed/len(subjects):.1f}s/subject avg)")

    # ─── Collect results ───
    actr_results = {}
    fear_results = {}
    rows = []

    for subj, n_test, facts, actr, fear in results:
        actr_results[int(subj)] = actr
        fear_results[int(subj)] = fear

        aic_a = 2 * actr['n_params'] + 2 * actr['nll']
        aic_f = 2 * fear['n_params'] + 2 * fear['nll']
        bic_a = actr['n_params'] * np.log(n_test) + 2 * actr['nll']
        bic_f = fear['n_params'] * np.log(n_test) + 2 * fear['nll']

        rows.append({
            'subj': subj, 'n_test': n_test,
            'nll_actr': actr['nll'],          'nll_fear': fear['nll'],
            'n_params_actr': actr['n_params'], 'n_params_fear': fear['n_params'],
            'aic_actr': aic_a,                'aic_fear': aic_f,
            'bic_actr': bic_a,                'bic_fear': bic_f,
            'delta_aic': aic_f - aic_a,
            'delta_bic': bic_f - bic_a,
            'c': actr['c'], 'phi': actr['phi'],
            'd': fear['d'], 'w1': fear['w1'],
            't0_actr': actr['t0'], 'F_actr': actr['F'],
            't0_fear': fear['t0'], 'F_fear': fear['F'],
            'converged_actr': actr['converged'],
            'converged_fear': fear['converged'],
        })

    summary = pd.DataFrame(rows).sort_values('subj')

    # ─── Save outputs ───
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_summary_by_subject.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'actr_results.json'), 'w') as fp:
        json.dump(actr_results, fp, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'fear_results.json'), 'w') as fp:
        json.dump(fear_results, fp, indent=2)

    # ─── Print summary ───
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary[['subj','nll_actr','nll_fear','delta_aic','delta_bic']].to_string(index=False))
    print(f"\nMean ΔAIC (FE - ACT-R): {summary['delta_aic'].mean():.2f}  (negative = FE-ACT-R better)")
    print(f"Mean ΔBIC (FE - ACT-R): {summary['delta_bic'].mean():.2f}")
    print(f"FE-ACT-R wins AIC: {(summary['delta_aic'] < 0).sum()}/{len(subjects)} subjects")
    print(f"FE-ACT-R wins BIC: {(summary['delta_bic'] < 0).sum()}/{len(subjects)} subjects")
    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
