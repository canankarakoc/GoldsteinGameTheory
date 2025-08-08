"""
Goldstein Evolutionary Game Theory Simulation

This script cleanly separates the simulation logic into defined scenarios:
- EI (Evolved Interaction) for Acute and Chronic
- ES (Evolved Strategy) for Acute and Chronic
- Mixed Strategy: ES with fixed host or fixed pathogen

Each simulation block is self-contained for easier debugging and interpretability.
"""

import numpy as np
import pandas as pd
import os
import random

np.random.seed(42)
random.seed(42)

# === MODEL PARAMETERS ===
beta = 1
d_0 = 0.1
m_c = 0.1
m_v = 1
epsilon = 1e-4
mutation_step_c_v = 0.1
mutation_step_theta = 0.314

gamma = 0.01
Ne_H = 10_000
Ne_P = 1_000_000
num_mutations = 10_000
max_dwell_time = 100
max_evo_time = 1_010_000

# === MODEL FUNCTIONS ===
def d_acute(c, v):
    return d_0 + m_c * (1 + epsilon) * c / (1 + epsilon - c) + m_v * (1 + epsilon) * v / (1 + epsilon - v)

def r_acute(v):
    return v ** beta

def f_H_acute(c, v):
    d = d_acute(c, v)
    return c / (d + c)

def f_P_acute(c, v):
    d = d_acute(c, v)
    r = r_acute(v)
    return r / (d + c)

def d_chronic(c, v):
    return d_0 + m_c * (1 + epsilon) * c / (1 + epsilon - c) + (1 - c) * m_v * (1 + epsilon) * v / (1 + epsilon - v)

def r_chronic(c, v):
    return v ** beta * (1 - c)

def f_H_chronic(c, v):
    d = d_chronic(c, v)
    return 1 / d

def f_P_chronic(c, v):
    r = r_chronic(c, v)
    d = d_chronic(c, v)
    return r / d

# === FIXATION + EVOLUTION HELPERS ===
def fixation_prob(s, Ne):
    if not np.isfinite(s):
        return 1 / Ne
    if abs(s) < 0.005:
        return 2 / Ne  # allow more neutral mutations to fix
    try:
        num = 1 - np.exp(-2 * s)
        denom = 1 - np.exp(-4 * Ne * s) if -4 * Ne * s > -700 else 0.0
        return num / denom if denom != 0 else 1 / Ne
    except:
        return 1 / Ne

def safe_choice(mutations, rates, evo_time):
    rates = np.array(rates, dtype=np.float64)
    total_rate = np.sum(rates)
    if total_rate == 0 or not np.isfinite(total_rate):
        return None, evo_time + 1, 0
    rates /= total_rate
    if not np.all(np.isfinite(rates)):
        return None, evo_time + 1, 0
    chosen_idx = np.random.choice(len(mutations), p=rates)
    dwell_time = min(np.random.exponential(1 / total_rate), max_dwell_time)
    return chosen_idx, evo_time + dwell_time, dwell_time

def find_equilibrium_ES(c_0, theta_c, v_0, theta_v, tol=1e-6, max_iter=100):
    theta_c = np.clip(theta_c, -1.4, 1.4)
    theta_v = np.clip(theta_v, -1.4, 1.4)
    c, v = 0.5, 0.5
    for _ in range(max_iter):
        c_new = np.clip(c_0 + np.tan(theta_c) * v, 0, 1)
        v_new = np.clip(v_0 + np.tan(theta_v) * c_new, 0, 1)
        if abs(c_new - c) < tol and abs(v_new - v) < tol:
            break
        c, v = c_new, v_new
    return c_new, v_new

# === EI SIMULATION ===
def simulate_EI(f_H, f_P, label="acute"):
    c, v = 0.5, 0.5
    ch, cp, time_now = 0, 0, 0.0
    neutral_host = selected_host = neutral_pathogen = selected_pathogen = 0
    res = []

    while time_now < max_evo_time:
        muts, rates = [], []

        # --- Host mutation proposals
        for _ in range(int(num_mutations * gamma)):
            delta = np.random.normal(0, mutation_step_c_v)
            c_new = np.clip(c + delta, 0, 0.999)
            f_old = f_H(c, v)
            f_new = f_H(c_new, v)
            s = (f_new - f_old) / f_old if f_old > 0 else 0
            if abs(s) < 0.01:
                neutral_host += 1
            else:
                selected_host += 1
            muts.append(('host', delta))
            rates.append(fixation_prob(s, Ne_H))

        # --- Pathogen mutation proposals
        for _ in range(int(num_mutations * (1 - gamma))):
            delta = np.random.normal(0, mutation_step_c_v)
            v_new = np.clip(v + delta, 0, 0.999)
            f_old = f_P(c, v)
            f_new = f_P(c, v_new)
            s = (f_new - f_old) / f_old if f_old > 0 else 0
            if abs(s) < 0.01:
                neutral_pathogen += 1
            else:
                selected_pathogen += 1
            muts.append(('pathogen', delta))
            rates.append(fixation_prob(s, Ne_P))

        # --- Mutation selection + time update (safe)
        idx, time_now, dwell_time = safe_choice(muts, rates, time_now)
        if idx is None:
            continue

        kind, delta = muts[idx]
        if kind == 'host':
            c = np.clip(c + delta, 0, 0.999)
            ch += 1
        else:
            v = np.clip(v + delta, 0, 0.999)
            cp += 1

        fH = f_H(c, v)
        fP = f_P(c, v)

        print(f"[EI {label.upper()}] t={time_now:.2f}, c={c:.3f}, v={v:.3f}, f_H={fH:.2e}, f_P={fP:.2e}")

        res.append({
            "time": time_now,
            "c": c,
            "v": v,
            "f_H": fH,
            "f_P": fP,
            "dwell_time": dwell_time,
            "omega_H": (ch / (time_now + 1e-6)) / (1 / Ne_H),
            "omega_P": (cp / (time_now + 1e-6)) / (1 / Ne_P)
        })

    df = pd.DataFrame(res)
    df.to_csv(f"ei_{label}.csv", index=False)
    summary = {
        "neutral_host": neutral_host,
        "selected_host": selected_host,
        "neutral_pathogen": neutral_pathogen,
        "selected_pathogen": selected_pathogen
    }
    return df, summary


# === ES SIMULATION ===
def simulate_ES(f_H, f_P, label="acute", fixed_host=False, fixed_pathogen=False):
    # Randomize initial starting point for more exploration
    c_0 = np.clip(np.random.uniform(0.2, 0.8), 0, 0.999)
    v_0 = np.clip(np.random.uniform(0.2, 0.8), 0, 0.999)
    theta_c = np.random.uniform(-0.5, 0.5)
    theta_v = np.random.uniform(-0.5, 0.5)

    res, ch, cp, time_now = [], 0, 0, 0.0
    neutral_host = selected_host = neutral_pathogen = selected_pathogen = 0

    while time_now < max_evo_time:
        muts, rates = [], []

        # Occasional random jumps to escape flat equilibria
        if np.random.rand() < 0.01:
            theta_c += np.random.normal(0, 0.3)
            theta_v += np.random.normal(0, 0.3)

        # --- Host mutations: c_0 or theta_c
        if not fixed_host:
            for _ in range(int(num_mutations * gamma)):
                param = np.random.choice([0, 1])
                delta = np.random.normal(0, mutation_step_theta if param else mutation_step_c_v)
                c0n, tcn = c_0, theta_c
                if param == 0: c0n = np.clip(c0n + delta, 0, 0.999)
                else: tcn += delta

                c_old, v_old = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
                c_new, v_new = find_equilibrium_ES(c0n, tcn, v_0, theta_v)
                f_old = f_H(c_old, v_old)
                f_new = f_H(c_new, v_new)
                s = (f_new - f_old) / f_old if f_old > 0 else 0
                if abs(s) < 0.01:
                    neutral_host += 1
                else:
                    selected_host += 1

                muts.append(('host', param, delta))
                rates.append(fixation_prob(s, Ne_H))

        # --- Pathogen mutations: v_0 or theta_v
        if not fixed_pathogen:
            for _ in range(int(num_mutations * (1 - gamma))):
                param = np.random.choice([0, 1])
                delta = np.random.normal(0, mutation_step_theta if param else mutation_step_c_v)
                v0n, tvn = v_0, theta_v
                if param == 0: v0n = np.clip(v0n + delta, 0, 0.999)
                else: tvn += delta

                c_old, v_old = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
                c_new, v_new = find_equilibrium_ES(c_0, theta_c, v0n, tvn)
                f_old = f_P(c_old, v_old)
                f_new = f_P(c_new, v_new)
                s = (f_new - f_old) / f_old if f_old > 0 else 0
                if abs(s) < 0.01:
                    neutral_pathogen += 1
                else:
                    selected_pathogen += 1

                muts.append(('pathogen', param, delta))
                rates.append(fixation_prob(s, Ne_P))

        # --- Choose mutation to fix
        idx, time_now, dwell_time = safe_choice(muts, rates, time_now)
        if idx is None:
            continue

        kind, param, delta = muts[idx]
        if kind == 'host' and not fixed_host:
            if param == 0: c_0 = np.clip(c_0 + delta, 0, 0.999)
            else: theta_c += delta
            ch += 1
        elif kind == 'pathogen' and not fixed_pathogen:
            if param == 0: v_0 = np.clip(v_0 + delta, 0, 0.999)
            else: theta_v += delta
            cp += 1

        c, v = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
        fH = f_H(c, v)
        fP = f_P(c, v)

        print(f"[ES {label.upper()}] t={time_now:.2f}, c={c:.3f}, v={v:.3f}, f_H={fH:.2e}, f_P={fP:.2e}")

        res.append({
            "time": time_now,
            "c": c,
            "v": v,
            "f_H": fH,
            "f_P": fP,
            "dwell_time": dwell_time,
            "omega_H": (ch / (time_now + 1e-6)) / (1 / Ne_H),
            "omega_P": (cp / (time_now + 1e-6)) / (1 / Ne_P),
            "theta_c": theta_c,
            "theta_v": theta_v,
            "c_0": c_0,
            "v_0": v_0
        })

    df = pd.DataFrame(res)
    df.to_csv(f"es_{label}.csv", index=False)
    summary = {
        "neutral_host": neutral_host,
        "selected_host": selected_host,
        "neutral_pathogen": neutral_pathogen,
        "selected_pathogen": selected_pathogen
    }
    return df, summary


# === MAIN BLOCK ===
if __name__ == "__main__":
    os.chdir("results")

    # === EI SCENARIOS ===
    #df_ei_acute, summary_ei_acute = simulate_EI(f_H_acute, f_P_acute, label="ei_acute")
    #df_ei_chronic, summary_ei_chronic = simulate_EI(f_H_chronic, f_P_chronic, label="ei_chronic")

    # === ES FULLY EVOLVING SCENARIOS ===
    #df_es_acute, summary_es_acute = simulate_ES(f_H_acute, f_P_acute, label="es_acute_full")
    df_es_chronic, summary_es_chronic = simulate_ES(f_H_chronic, f_P_chronic, label="es_chronic_full")

    # === ES WITH FIXED HOST  AND PATHOGEN ===
    df_es_acute_fixedhost, summary_es_acute_fixedhost = simulate_ES(f_H_acute, f_P_acute, label="es_acute_fixed_host")
    df_es_acute_fixedpathogen, summary_es_acute_fixedpathogen = simulate_ES(f_H_acute, f_P_acute, label="es_acute_fixed_pathogen")
    df_es_chronic_fixedhost, summary_es_chronic_fixedhost = simulate_ES(f_H_chronic, f_P_chronic, label="es_chronic_fixed_host")
    df_es_chronic_fixedpathogen, summary_es_chronic_fixedpathogen = simulate_ES(f_H_chronic, f_P_chronic, label="es_chronic_fixed_pathogen")


    #print("\nSummary Statistics:")
    #print("=== EI ===")
    #print("  Acute :", summary_ei_acute)
    #print("  Chronic:", summary_ei_chronic)

   #print("\n=== ES Full ===")
    #print("  Acute :", summary_es_acute)
    print("  Chronic:", summary_es_chronic)

    print("\n=== ES Fixed Host ===")
    print("  Acute :", summary_es_acute_fixedhost)
    print("  Chronic:", summary_es_chronic_fixedhost)

    print("\n=== ES Fixed Pathogen ===")
    print("  Acute :", summary_es_acute_fixedpathogen)
    print("  Chronic:", summary_es_chronic_fixedpathogen)

    # === SAVE SUMMARY FILE ===
    all_summaries = {
        #"ei_acute": summary_ei_acute,
        #"ei_chronic": summary_ei_chronic,
        #"es_acute_full": summary_es_acute,
        "es_chronic_full": summary_es_chronic,
        "es_acute_fixed_host": summary_es_acute_fixedhost,
        "es_chronic_fixed_host": summary_es_chronic_fixedhost,
        "es_acute_fixed_pathogen": summary_es_acute_fixedpathogen,
        "es_chronic_fixed_pathogen": summary_es_chronic_fixedpathogen,
    }

    import json
    with open("simulation_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=4)