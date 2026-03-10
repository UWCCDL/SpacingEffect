"""
Model comparison: ACT-R (Pavlik & Anderson, 2005) vs FE-ACT-R
Fitted to the SlimStampen dataset.

Flexible parameter configuration: each parameter can be fixed (constant)
or free (optimized), and free parameters can be estimated at the level of
'subject' or 'fact' (subject x fact).

USAGE:
    python fit_models_slimstampen.py

Edit DATA_PATH, OUTPUT_DIR, N_RESTARTS, N_WORKERS, and the PARAMS dicts below.
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
DATA_PATH  = 'newdata.csv'
OUTPUT_DIR = '.'
N_RESTARTS = 15
N_WORKERS  = None  # None = all available cores

# Parameter config: either a constant value, or a dict with 'bounds' and 'level'.
# 'level' can be 'subject' or 'fact' (subject x fact).
# t0 bounds are overridden dynamically: upper = 0.95 * min(RT) for the subject.

ACTR_PARAMS = {
    'c':   {'bounds': (0, 1),   'level': 'subject'},
    #'c' : 0.5,
    'phi': {'bounds': (0.1, 0.75),   'level': 'fact'},
    #'t0': 0.3,
    't0': {'bounds':(0.3, 0.8), 'level': 'subject'},
    'F': 1,
    #'F': {'bounds': (0.5, 1.5), 'level': 'subject'},
    'tau': -0.8,
    #'tau': {'bounds':(-2, 0), 'level':'subject'},
    #'s':   0.25,
    's': {'bounds': (0.1, 1),   'level': 'subject'},
}

FEAR_PARAMS = {
    #'d':   {'bounds': (0.1, 0.75),   'level': 'subject'},
    'd':   {'bounds': (0.1, 0.75),   'level': 'subject'},
    'w1':  {'bounds': (0, 5),  'level': 'fact'}, # 0-10
    #'w1': 2.6,
    # 't0': 0.3,
    't0': {'bounds':(0.3, 0.8), 'level': 'subject'},
    #'F':   {'bounds': (0.5, 1.5), 'level': 'subject'},
    'F': 1,
    'tau': -0.8,
    #'tau': {'bounds':(-2, 0), 'level':'subject'},
    #'s':   0.25,
    's': {'bounds': (0.1, 1),   'level': 'subject'},
}


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def beta_from_s(s):
    """Shape parameter of the log-logistic RT distribution."""
    return np.sqrt(3) / (np.pi * s)


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

def compute_activation_fear(enc_times, t_query, d, w1, tau, s):
    n = len(enc_times)
    if n == 0:
        return np.nan
    weights = np.empty(n)
    weights[0] = w1
    for i in range(1, n):
        dt = np.maximum(enc_times[i] - enc_times[:i], 1e-10)
        A  = np.log(max(np.sum(weights[:i] * (dt ** (-d))), 1e-10))
        p  = float(sigmoid((A - tau) / s))
        p  = np.clip(p, 1e-10, 1 - 1e-10)
        weights[i] = -np.log(p)
    dt_q = np.maximum(t_query - enc_times, 1e-10)
    return np.log(max(np.sum(weights * (dt_q ** (-d))), 1e-10))


# ─────────────────────────────────────────────
# LOG-LIKELIHOOD COMPONENTS
# ─────────────────────────────────────────────

def log_lik_accuracy(A, is_correct, tau, s):
    x = (A - tau) / s
    return -np.log1p(np.exp(-x)) if is_correct else -np.log1p(np.exp(x))

def log_lik_rt(rt_shifted, A, F, s):
    beta  = beta_from_s(s)
    alpha = F * np.exp(-A)
    if alpha <= 0 or rt_shifted <= 0:
        return -1e6
    z = rt_shifted / alpha
    return (np.log(beta) - np.log(alpha)
            + (beta - 1) * np.log(z)
            - 2 * np.log1p(z ** beta))


# ─────────────────────────────────────────────
# SEQUENCE PRE-PROCESSING
# ─────────────────────────────────────────────

def build_fact_sequences(fact_df):
    """Convert a fact's trials into a list of (enc_snapshot, t_query, is_correct, rt_s)."""
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

def nll_fact_actr(queries, c, phi, t0, F, tau, s):
    total = 0.0
    for enc_snap, tq, correct, rt_s in queries:
        A = compute_activation_actr(enc_snap, tq, c, phi)
        if not np.isfinite(A):
            total += 1e6; continue
        total -= log_lik_accuracy(A, correct, tau, s) + log_lik_rt(rt_s - t0, A, F, s)
    return total

def nll_fact_fear(queries, d, w1, t0, F, tau, s):
    total = 0.0
    for enc_snap, tq, correct, rt_s in queries:
        A = compute_activation_fear(enc_snap, tq, d, w1, tau, s)
        if not np.isfinite(A):
            total += 1e6; continue
        total -= log_lik_accuracy(A, correct, tau, s) + log_lik_rt(rt_s - t0, A, F, s)
    return total


# ─────────────────────────────────────────────
# PARAMETER SYSTEM
# ─────────────────────────────────────────────

def is_fixed(spec):
    return not isinstance(spec, dict)

def get_group_keys(subj_data, level):
    """Return sorted list of unique group keys for a given level."""
    if level == 'subject':
        return ['subject']
    elif level == 'fact':
        return sorted(subj_data['fact_uid'].unique())

def build_param_index(param_config, subj_data):
    """
    Build a flat list of (param_name, group_key, lo, hi) for all free parameters,
    and a lookup dict: (param_name, group_key) -> index in flat vector.
    """
    entries = []
    index   = {}
    for pname, spec in param_config.items():
        if is_fixed(spec):
            continue
        level  = spec['level']
        bounds = spec['bounds']
        keys   = get_group_keys(subj_data, level)
        for key in keys:
            idx = len(entries)
            index[(pname, key)] = idx
            entries.append((pname, key, bounds[0], bounds[1]))
    return entries, index

def resolve_params(param_config, param_index, x, fact_uid):
    """Resolve all parameters for a specific fact."""
    resolved = {}
    for pname, spec in param_config.items():
        if is_fixed(spec):
            resolved[pname] = float(spec)
        else:
            level = spec['level']
            key   = 'subject' if level == 'subject' else fact_uid
            resolved[pname] = x[param_index[(pname, key)]]
    return resolved

def compute_t0_upper(subj_data, level, group_key):
    """Compute dynamic t0 upper bound = 0.95 * min(RT) for the relevant group."""
    test = subj_data[subj_data['type'] == 'test']
    subset = test if level == 'subject' else test[test['fact_uid'] == group_key]
    if len(subset) == 0:
        return 1.0
    return min(0.95 * subset['RT_s'].min(), 1.0)

def apply_t0_bounds(entries, param_config, subj_data):
    """Override t0 upper bounds dynamically based on min RT per group."""
    if 't0' not in param_config or is_fixed(param_config['t0']):
        return entries
    level = param_config['t0']['level']
    updated = []
    for pname, key, lo, hi in entries:
        if pname == 't0':
            hi = compute_t0_upper(subj_data, level, key)
            lo = min(lo, hi * 0.5)
        updated.append((pname, key, lo, hi))
    return updated


# ─────────────────────────────────────────────
# OBJECTIVE FUNCTION (top-level for pickling)
# ─────────────────────────────────────────────

def _objective(args):
    x, seqs, facts, param_config, param_index, entries, model = args
    lowers = np.array([e[2] for e in entries])
    uppers = np.array([e[3] for e in entries])
    x = np.clip(x, lowers, uppers)

    nll_fn = nll_fact_actr if model == 'actr' else nll_fact_fear
    pnames = list(param_config.keys())
    total  = 0.0
    for fact_uid in facts:
        p    = resolve_params(param_config, param_index, x, fact_uid)
        vals = [p[n] for n in pnames]
        total += nll_fn(seqs[fact_uid], *vals)
    return total if np.isfinite(total) else 1e9


# ─────────────────────────────────────────────
# PER-PARTICIPANT FITTING
# ─────────────────────────────────────────────

def _powell(obj, x0, bounds_list, coarse=False):
    opts = ({'maxiter': 10000, 'maxfev': 800,    'ftol': 1e-3, 'xtol': 1e-3} if coarse else
            {'maxiter': 10000, 'maxfev': 100000,  'ftol': 1e-6, 'xtol': 1e-6})
    scipy_bounds = [(lo, hi) for _, _, lo, hi in bounds_list]
    return minimize(obj, x0, method='Powell', bounds=scipy_bounds, options=opts)

def fit_participant(subj_data, facts, model, n_restarts=N_RESTARTS):
    param_config = ACTR_PARAMS if model == 'actr' else FEAR_PARAMS
    entries, param_index = build_param_index(param_config, subj_data)
    entries = apply_t0_bounds(entries, param_config, subj_data)

    lowers = np.array([e[2] for e in entries])
    uppers = np.array([e[3] for e in entries])

    seqs = {fuid: build_fact_sequences(subj_data[subj_data['fact_uid'] == fuid])
            for fuid in facts}

    obj = lambda x: _objective((x, seqs, facts, param_config, param_index, entries, model))

    rng  = np.random.default_rng(42)
    best = None
    for _ in range(n_restarts):
        x0  = lowers + rng.random(len(entries)) * (uppers - lowers)
        res = _powell(obj, x0, entries, coarse=True)
        if best is None or res.fun < best.fun:
            best = res
    best = _powell(obj, best.x, entries, coarse=False)
    return best, entries, param_index


# ─────────────────────────────────────────────
# TRIAL-LEVEL PREDICTIONS
# ─────────────────────────────────────────────

def predict_trials(subj_data, facts, param_config, param_index, x, model):
    """
    For every test trial, compute predicted P(correct) and predicted median RT.
    Returns a DataFrame with one row per test trial.
    """
    x      = np.clip(x, [0.0] * len(x), [1e9] * len(x))
    prefix = 'actr' if model == 'actr' else 'fear'
    pnames = list(param_config.keys())

    records = []
    for fact_uid in facts:
        p   = resolve_params(param_config, param_index, x, fact_uid)
        tau, s = p['tau'], p['s']
        t0,  F = p['t0'],  p['F']

        fact_df  = subj_data[subj_data['fact_uid'] == fact_uid].sort_values('trial')
        enc_list = []

        for _, row in fact_df.iterrows():
            t = row['time']
            if row['type'] == 'study':
                enc_list.append(t)
                continue

            enc_snap = np.array(enc_list, dtype=float)
            if model == 'actr':
                A = compute_activation_actr(enc_snap, t, p['c'], p['phi'])
            else:
                A = compute_activation_fear(enc_snap, t, p['d'], p['w1'], tau, s)

            if np.isfinite(A):
                p_correct = float(sigmoid((A - tau) / s))
                pred_rt   = t0 + F * np.exp(-A)
            else:
                p_correct = np.nan
                pred_rt   = np.nan

            rec = {
                'fact_uid':   str(fact_uid),
                'trial':      int(row['trial']),
                f'{prefix}_A':          round(A, 6) if np.isfinite(A) else np.nan,
                f'{prefix}_p_correct':  round(p_correct, 6) if np.isfinite(p_correct) else np.nan,
                f'{prefix}_pred_RT':    round(pred_rt, 6) if np.isfinite(pred_rt) else np.nan,
            }
            for pname, val in p.items():
                rec[f'{prefix}_{pname}'] = round(float(val), 6)

            records.append(rec)
            enc_list.append(t)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# WORKER (top-level for pickling)
# ─────────────────────────────────────────────

def _fit_subject(args):
    subj, subj_data, n_restarts = args

    subj_data = subj_data.copy()
    subj_data['fact_uid'] = list(zip(subj_data['subj'], subj_data['item']))
    facts = sorted(subj_data['fact_uid'].unique())
    n_test = len(subj_data[subj_data['type'] == 'test'])

    res_a, entries_a, idx_a = fit_participant(subj_data, facts, 'actr', n_restarts)
    res_f, entries_f, idx_f = fit_participant(subj_data, facts, 'fear', n_restarts)

    def pack(res, entries, param_index, param_config):
        x   = np.clip(res.x, [e[2] for e in entries], [e[3] for e in entries])
        out = {'nll': float(res.fun), 'n_params': len(entries),
               'converged': bool(res.success), 'params': {}}
        for pname, spec in param_config.items():
            if is_fixed(spec):
                out['params'][pname] = float(spec)
            else:
                level = spec['level']
                keys  = get_group_keys(subj_data, level)
                if level == 'subject':
                    out['params'][pname] = float(x[param_index[(pname, 'subject')]])
                else:
                    out['params'][pname] = [
                        {'key': list(k), 'value': float(x[param_index[(pname, k)]])}
                        for k in keys
                    ]
        return out

    actr = pack(res_a, entries_a, idx_a, ACTR_PARAMS)
    fear = pack(res_f, entries_f, idx_f, FEAR_PARAMS)

    x_a = np.clip(res_a.x, [e[2] for e in entries_a], [e[3] for e in entries_a])
    x_f = np.clip(res_f.x, [e[2] for e in entries_f], [e[3] for e in entries_f])
    pred_a = predict_trials(subj_data, facts, ACTR_PARAMS, idx_a, x_a, 'actr')
    pred_f = predict_trials(subj_data, facts, FEAR_PARAMS, idx_f, x_f, 'fear')
    preds  = pred_a.merge(pred_f, on=['fact_uid', 'trial'], how='outer')

    return subj, n_test, actr, fear, preds


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

    for model, cfg in [('ACT-R', ACTR_PARAMS), ('FE-ACT-R', FEAR_PARAMS)]:
        free  = {k: v for k, v in cfg.items() if not is_fixed(v)}
        fixed = {k: v for k, v in cfg.items() if is_fixed(v)}
        print(f"\n{model} free:  " + ', '.join(f"{k} [{v['level']}]" for k, v in free.items()))
        if fixed:
            print(f"{model} fixed: " + ', '.join(f"{k}={v}" for k, v in fixed.items()))

    job_args = [(subj, df[df['subj'] == subj].copy(), N_RESTARTS)
                for subj in subjects]

    t_start = time.time()
    print(f"\nFitting all subjects in parallel ({n_workers} cores)...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(_fit_subject, job_args)

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.0f}s ({elapsed/len(subjects):.1f}s/subject avg)")

    actr_results = {}
    fear_results = {}
    all_preds    = []
    rows         = []

    for subj, n_test, actr, fear, preds in results:
        actr_results[int(subj)] = actr
        fear_results[int(subj)] = fear
        preds['subj'] = subj
        all_preds.append(preds)

        aic_a = 2 * actr['n_params'] + 2 * actr['nll']
        aic_f = 2 * fear['n_params'] + 2 * fear['nll']
        bic_a = actr['n_params'] * np.log(n_test) + 2 * actr['nll']
        bic_f = fear['n_params'] * np.log(n_test) + 2 * fear['nll']

        row = {
            'subj': subj, 'n_test': n_test,
            'nll_actr': actr['nll'],           'nll_fear': fear['nll'],
            'n_params_actr': actr['n_params'],  'n_params_fear': fear['n_params'],
            'aic_actr': aic_a,                  'aic_fear': aic_f,
            'bic_actr': bic_a,                  'bic_fear': bic_f,
            'delta_aic': aic_f - aic_a,
            'delta_bic': bic_f - bic_a,
            'converged_actr': actr['converged'],
            'converged_fear': fear['converged'],
        }
        for pname, val in actr['params'].items():
            if isinstance(val, float):
                row[f'actr_{pname}'] = val
        for pname, val in fear['params'].items():
            if isinstance(val, float):
                row[f'fear_{pname}'] = val
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values('subj')

    # Merge trial predictions with original test data
    trial_preds = pd.concat(all_preds, ignore_index=True)
    df['fact_uid'] = df.apply(lambda r: str((r['subj'], r['item'])), axis=1)
    test_df = df[df['type'] == 'test'].copy()
    trial_output = test_df.merge(trial_preds, on=['subj', 'fact_uid', 'trial'], how='left')
    trial_output = trial_output.drop(columns=['fact_uid'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'slimstampen_model_comparison_summary.csv'), index=False)
    trial_output.to_csv(os.path.join(OUTPUT_DIR, 'slimstampen_trial_predictions.csv'), index=False)
    with open(os.path.join(OUTPUT_DIR, 'slimstampen_actr_results.json'), 'w') as fp:
        json.dump(actr_results, fp, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'slimstampen_fear_results.json'), 'w') as fp:
        json.dump(fear_results, fp, indent=2)

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
