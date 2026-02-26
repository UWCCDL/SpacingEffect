"""
Model comparison: ACT-R (Pavlik & Anderson, 2005) vs FE-ACT-R
Uses Powell optimizer with random restarts.

USAGE:
    python fit_models.py

Edit DATA_PATH and OUTPUT_DIR below to match your setup.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import json
import time
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH  = 'slimstampen_all_trials.csv'  # path to your CSV data file
OUTPUT_DIR = '.'                            # directory for output files
N_RESTARTS = 15                             # random restarts per subject/model

# ─────────────────────────────────────────────
# FIXED PARAMETERS
# ─────────────────────────────────────────────
TAU  = -0.8
S    = 0.25
BETA = np.sqrt(3) / (np.pi * S)   # ≈ 2.21, fixed

# ─────────────────────────────────────────────
# PARAMETER BOUNDS
# ─────────────────────────────────────────────
ACTR_PP_BOUNDS = [(0, 1), (0.3, 1), (0.5, 2)]   # c, t0, F
ACTR_PF_BOUNDS = [(0, 1)]                         # phi per fact

FEAR_PP_BOUNDS = [(0, 1), (0.3, 1), (0.5, 2)]   # d, t0, F
FEAR_PF_BOUNDS = [(0, 10)]                        # w1 per fact


# ─────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def compute_activation_actr(enc_times, t_query, c, phi):
    """
    Compute ACT-R activation at t_query given encoding times enc_times.
    Uses a lower-triangular dt matrix to avoid repeated slicing.
    """
    n = len(enc_times)
    if n == 0:
        return np.nan

    decay_rates = np.empty(n)
    decay_rates[0] = phi

    if n > 1:
        # Lower-triangular matrix of time differences: dt[i,j] = enc_times[i] - enc_times[j], j < i
        et = enc_times
        for i in range(1, n):
            dt = et[i] - et[:i]
            np.maximum(dt, 1e-10, out=dt)
            s = np.dot(dt ** (-decay_rates[:i]), np.ones(i))
            A = np.log(s) if s > 1e-10 else -23.0
            decay_rates[i] = c * np.exp(A) + phi

    dt_q = t_query - enc_times
    np.maximum(dt_q, 1e-10, out=dt_q)
    s = np.dot(dt_q ** (-decay_rates), np.ones(n))
    return np.log(s) if s > 1e-10 else -23.0


def compute_activation_fear(enc_times, t_query, d, w1):
    """
    Compute FE-ACT-R activation at t_query given encoding times enc_times.
    """
    n = len(enc_times)
    if n == 0:
        return np.nan

    weights = np.empty(n)
    weights[0] = w1

    if n > 1:
        neg_d = -d
        et = enc_times
        for i in range(1, n):
            dt = et[i] - et[:i]
            np.maximum(dt, 1e-10, out=dt)
            s = np.dot(weights[:i], dt ** neg_d)
            A = np.log(s) if s > 1e-10 else -23.0
            p = sigmoid((A - TAU) / S)
            if p < 1e-10: p = 1e-10
            elif p > 1 - 1e-10: p = 1 - 1e-10
            weights[i] = -np.log(p)

    dt_q = t_query - enc_times
    np.maximum(dt_q, 1e-10, out=dt_q)
    s = np.dot(weights, dt_q ** (-d))
    return np.log(s) if s > 1e-10 else -23.0


# ─────────────────────────────────────────────
# LOG-LIKELIHOOD COMPONENTS
# ─────────────────────────────────────────────

def log_lik_accuracy(A, is_correct):
    x = (A - TAU) / S
    return np.where(is_correct, -np.log1p(np.exp(-x)), -np.log1p(np.exp(x)))

def log_lik_rt(rt_shifted, A, F):
    alpha = F * np.exp(-A)
    alpha = np.maximum(alpha, 1e-10)
    valid = rt_shifted > 0
    z = np.where(valid, rt_shifted / alpha, 1.0)
    log_pdf = (np.log(BETA) - np.log(alpha)
               + (BETA - 1) * np.log(z)
               - 2 * np.log1p(z ** BETA))
    return np.where(valid, log_pdf, -1e6)


# ─────────────────────────────────────────────
# PER-FACT NLL
# ─────────────────────────────────────────────

def build_fact_sequences(fact_df):
    """Pre-process a fact's trials into sequences for fast NLL computation."""
    rows = fact_df.sort_values('trial').reset_index(drop=True)
    enc_list   = []
    queries    = []   # (enc_snapshot, t_query, is_correct, rt_s)

    for _, row in rows.iterrows():
        t = row['time']
        if row['type'] == 'study':
            enc_list.append(t)
        else:
            queries.append((
                np.array(enc_list, dtype=float),
                float(t),
                bool(row['isCorrect']),
                float(row['RT_s'])
            ))
            enc_list.append(t)
    return queries


def nll_fact_actr(queries, c, phi, t0, F):
    if not queries:
        return 0.0
    total = 0.0
    for enc_snap, tq, correct, rt_s in queries:
        A = compute_activation_actr(enc_snap, tq, c, phi)
        if not np.isfinite(A):
            total += 1e6
            continue
        rt_shifted = rt_s - t0
        ll = log_lik_accuracy(A, correct) + log_lik_rt(rt_shifted, A, F)
        total -= float(ll)
    return total


def nll_fact_fear(queries, d, w1, t0, F):
    if not queries:
        return 0.0
    total = 0.0
    for enc_snap, tq, correct, rt_s in queries:
        A = compute_activation_fear(enc_snap, tq, d, w1)
        if not np.isfinite(A):
            total += 1e6
            continue
        rt_shifted = rt_s - t0
        ll = log_lik_accuracy(A, correct) + log_lik_rt(rt_shifted, A, F)
        total -= float(ll)
    return total


# ─────────────────────────────────────────────
# PER-PARTICIPANT FITTING
# ─────────────────────────────────────────────

def fit_actr_participant(subj_data, facts, n_restarts=10):
    n_facts = len(facts)
    bounds  = ACTR_PP_BOUNDS + ACTR_PF_BOUNDS * n_facts
    lowers  = np.array([b[0] for b in bounds])
    uppers  = np.array([b[1] for b in bounds])

    seqs = {fact: build_fact_sequences(subj_data[subj_data['item'] == fact])
            for fact in facts}

    def objective(params):
        params = np.clip(params, lowers, uppers)
        c, t0, F = params[0], params[1], params[2]
        phis = params[3:]
        total = sum(nll_fact_actr(seqs[fact], c, phis[k], t0, F)
                    for k, fact in enumerate(facts))
        return total if np.isfinite(total) else 1e9

    rng  = np.random.default_rng(42)
    best = None
    # Phase 1: coarse multi-start
    for _ in range(n_restarts):
        x0  = lowers + rng.random(len(bounds)) * (uppers - lowers)
        res = minimize(objective, x0, method='Powell',
                       options={'maxiter': 10000, 'maxfev': 800,
                                'ftol': 1e-3, 'xtol': 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    # Phase 2: polish from best
    best = minimize(objective, best.x, method='Powell',
                    options={'maxiter': 10000, 'maxfev': 10000,
                             'ftol': 1e-6, 'xtol': 1e-6})
    return best


def fit_fear_participant(subj_data, facts, n_restarts=10):
    n_facts = len(facts)
    bounds  = FEAR_PP_BOUNDS + FEAR_PF_BOUNDS * n_facts
    lowers  = np.array([b[0] for b in bounds])
    uppers  = np.array([b[1] for b in bounds])
    # params layout: [d, t0, F, w1_0, w1_1, ..., w1_{n-1}]

    seqs = {fact: build_fact_sequences(subj_data[subj_data['item'] == fact])
            for fact in facts}

    def objective(params):
        params = np.clip(params, lowers, uppers)
        d, t0, F = params[0], params[1], params[2]
        w1s = params[3:]
        total = sum(nll_fact_fear(seqs[fact], d, w1s[k], t0, F)
                    for k, fact in enumerate(facts))
        return total if np.isfinite(total) else 1e9

    rng  = np.random.default_rng(42)
    best = None
    # Phase 1: coarse multi-start
    for _ in range(n_restarts):
        x0  = lowers + rng.random(len(bounds)) * (uppers - lowers)
        res = minimize(objective, x0, method='Powell',
                       options={'maxiter': 10000, 'maxfev': 800,
                                'ftol': 1e-3, 'xtol': 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    # Phase 2: polish from best
    best = minimize(objective, best.x, method='Powell',
                    options={'maxiter': 10000, 'maxfev': 10000,
                             'ftol': 1e-6, 'xtol': 1e-6})
    return best


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

    subjects = sorted(df['subj'].unique())
    print(f"Subjects: {len(subjects)}")

    actr_results = {}
    fear_results = {}

    for i, subj in enumerate(subjects):
        subj_data = df[df['subj'] == subj]
        facts     = sorted(subj_data['item'].unique())
        n_test    = len(subj_data[subj_data['type'] == 'test'])
        print(f"\n[{i+1}/{len(subjects)}] Subject {subj} | {len(facts)} facts | {n_test} test trials")

        # --- ACT-R ---
        t_start = time.time()
        print(f"  Fitting ACT-R  ({3 + len(facts)} params)...", end='', flush=True)
        res_a = fit_actr_participant(subj_data, facts)
        dt_a  = time.time() - t_start
        actr_results[subj] = {
            'nll':      float(res_a.fun),
            'c':        float(res_a.x[0]),
            't0':       float(res_a.x[1]),
            'F':        float(res_a.x[2]),
            'phi':      {int(f): float(res_a.x[3+k]) for k, f in enumerate(facts)},
            'n_params': 3 + len(facts),
            'converged': bool(res_a.success)
        }
        print(f" NLL={res_a.fun:.2f} | c={res_a.x[0]:.3f}, t0={res_a.x[1]:.3f}, F={res_a.x[2]:.3f} [{dt_a:.0f}s]")

        # --- FE-ACT-R ---
        t_start = time.time()
        print(f"  Fitting FE-ACT-R ({2 + 2*len(facts)} params)...", end='', flush=True)
        res_f = fit_fear_participant(subj_data, facts)
        dt_f  = time.time() - t_start
        fear_results[subj] = {
            'nll':      float(res_f.fun),
            'd':        float(res_f.x[0]),
            't0':       float(res_f.x[1]),
            'F':        float(res_f.x[2]),
            'w1':       {int(f): float(res_f.x[3+k]) for k, f in enumerate(facts)},
            'n_params': 3 + len(facts),
            'converged': bool(res_f.success)
        }
        print(f" NLL={res_f.fun:.2f} | d={res_f.x[0]:.3f}, t0={res_f.x[1]:.3f}, F={res_f.x[2]:.3f} [{dt_f:.0f}s]")

    # ─── Summary ───
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    rows = []
    for subj in subjects:
        a = actr_results[subj]
        f = fear_results[subj]
        n_test = len(df[(df['subj'] == subj) & (df['type'] == 'test')])
        aic_a  = 2 * a['n_params'] + 2 * a['nll']
        aic_f  = 2 * f['n_params'] + 2 * f['nll']
        bic_a  = a['n_params'] * np.log(n_test) + 2 * a['nll']
        bic_f  = f['n_params'] * np.log(n_test) + 2 * f['nll']
        rows.append({
            'subj': subj, 'n_test': n_test,
            'nll_actr': a['nll'],        'nll_fear': f['nll'],
            'n_params_actr': a['n_params'], 'n_params_fear': f['n_params'],
            'aic_actr': aic_a,           'aic_fear': aic_f,
            'bic_actr': bic_a,           'bic_fear': bic_f,
            'delta_aic': aic_f - aic_a,
            'delta_bic': bic_f - bic_a,
            'd': f['d'],
            't0_actr': a['t0'], 'F_actr': a['F'],
            't0_fear': f['t0'], 'F_fear': f['F'],
            'converged_actr': a['converged'],
            'converged_fear': f['converged'],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison_summary.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'actr_results.json'), 'w') as fp:
        json.dump(actr_results, fp, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'fear_results.json'), 'w') as fp:
        json.dump(fear_results, fp, indent=2)

    print(summary[['subj','nll_actr','nll_fear','delta_aic','delta_bic']].to_string(index=False))
    print(f"\nMean ΔAIC (FE - ACT-R): {summary['delta_aic'].mean():.2f}  (negative = FE-ACT-R better)")
    print(f"Mean ΔBIC (FE - ACT-R): {summary['delta_bic'].mean():.2f}")
    print(f"FE-ACT-R wins AIC: {(summary['delta_aic'] < 0).sum()}/{len(subjects)} subjects")
    print(f"FE-ACT-R wins BIC: {(summary['delta_bic'] < 0).sum()}/{len(subjects)} subjects")
    print(f"\nResults saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
