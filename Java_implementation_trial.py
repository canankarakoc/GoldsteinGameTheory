
"""
Goldstein Evolutionary Game Theory Simulation 

This script cleanly separates the simulation logic into four defined scenarios:
1. EI - Acute
2. EI - Chronic
3. ES - Acute
4. ES - Chronic

Each simulation block is fully self-contained for easier debugging and interpretability.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import time
import random
import os 

np.random.seed(42)
random.seed(42)

# === MODEL PARAMETERS ===
nC = 0.1              # Host clearance cost multiplier
nV = 1                # Pathogen virulence cost multiplier
beta = 1              # Exponent in pathogen growth rate
d_0 = 0.1             # Baseline mortality

m_c = 0.1             # Chronic-specific clearance cost multiplier
m_v = 1               # Chronic-specific virulence cost multiplier

# Constants for mortality calculations
epsilon = 1.0 / 999.0
onePlusEpsilon = 1.0 + epsilon

# Simulation scale and granularity
Ne_H = 10_000              # Effective population size for host
Ne_P = 1_000_000           # Effective population size for pathogen
num_mutations = 10000      # Number of mutations per time step
gamma = 0.01               # Fraction of host mutations
max_dwell_time = 100      # Max allowed time for a substitution to occur
max_evo_time = 1_010_000      # Maximum simulation time

# === Mutation Step Parameters ===
stdDevMove = 0.1          # Mutation step size for trait movement
stdDevAngle = 0.314        # Angular step for strategy angle (theta) ~ 0.1π


def safe_div(x):
    return np.clip(x, 0.001, 0.999)  # or 0.05 to 0.95

# === MORTALITY FUNCTION (ACUTE ONLY) ===
def compute_mortality(c, v):
    c = safe_div(c)
    v = safe_div(v)
    clearance_term = nC * onePlusEpsilon * c / (onePlusEpsilon - c)
    virulence_term = nV * onePlusEpsilon * v / (onePlusEpsilon - v)
    return d_0 + clearance_term + virulence_term


# === TRANSMISSION FUNCTION (ACUTE ONLY) ===
def compute_transmission(v):
    return v ** beta

# === HOST FITNESS (ACUTE) ===
def f_H_acute(c, v):
    c = safe_div(c)
    v = safe_div(v)
    d = compute_mortality(c, v)
    return c / (c + d)


# === PATHOGEN FITNESS (ACUTE) ===
def f_P_acute(c, v):
    d = compute_mortality(c, v)
    t = compute_transmission(v)
    return t / (c + d)


# === MORTALITY FUNCTION (CHRONIC ONLY) ===
def d_chronic(c, v):
    c = safe_div(c)
    v = safe_div(v)
    clearance_term = m_c * onePlusEpsilon * c / (onePlusEpsilon - c)
    virulence_term = (1 - c) * m_v * onePlusEpsilon * v / (onePlusEpsilon - v)
    return d_0 + clearance_term + virulence_term


# === REPLICATION FUNCTION (CHRONIC ONLY) ===
def r_chronic(c, v):
    return (1 - c) * v ** beta


# === HOST FITNESS (CHRONIC) ===
def f_H_chronic(c, v):
    d = d_chronic(c, v)
    return 1.0 / d


# === PATHOGEN FITNESS (CHRONIC) ===
def f_P_chronic(c, v):
    d = d_chronic(c, v)
    r = r_chronic(c, v)
    return r / d


# === UTILITY FUNCTIONS ===

def gaussian_step_bins(n_steps=51, std_dev=0.1):
    """
    Create a weighted Gaussian-distributed list of step bins and their probabilities.
    Returns:
        steps: midpoints of Gaussian bins
        probs: normalized probability for each bin
    """
    percentiles = np.linspace(0, 1, n_steps + 1)
    z_vals = norm.ppf(percentiles)
    z_vals[0] = z_vals[1] * 10
    z_vals[-1] = z_vals[-2] * 10

    steps = 0.5 * (z_vals[1:] + z_vals[:-1])
    steps *= std_dev

    pdf_vals = norm.pdf(steps / std_dev)
    probs = pdf_vals / np.sum(pdf_vals)

    return steps, probs

def fixation_prob(s, Ne):
    """Numerically stable fixation probability function."""
    if not np.isfinite(s):
        return 1.0 / Ne
    if abs(s * Ne) < 1e-3:
        return 1.0 / Ne
    try:
        num = -2.0 * s
        denom = -4.0 * Ne * s
        return 2.0 * Ne * (1 - np.exp(num)) / (1 - np.exp(denom))
    except OverflowError:
        return 0.0 if s < 0 else 2.0 * Ne

def find_intersection(b_c, m_c, b_v, m_v, jitter=False):
    """Intersection of lines with optional jitter to avoid parallel lines."""
    denom = 1 - m_c * m_v
    if abs(denom) < 1e-10:
        if jitter:
            m_c += 1e-4 * np.random.randn()
            m_v += 1e-4 * np.random.randn()
            return find_intersection(b_c, m_c, b_v, m_v, jitter=False)
        else:
            return None
    v = (b_c + m_c * b_v) / denom
    c = b_v + m_v * v
    if not (0.001 <= v <= 0.999 and 0.001 <= c <= 0.999):
        return None
    return (c, v)

# Define mutation step bins + probabilities
move_step, move_prob = gaussian_step_bins(n_steps=51, std_dev=stdDevMove)
angle_step, angle_prob = gaussian_step_bins(n_steps=51, std_dev=stdDevAngle)


# ===============================
# === SIMULATION FUNCTIONS   ===
# ===============================

# === EI SIMULATIONS ===

def simulate_EI(f_H, f_P):
    """Simulates EI dynamics with trait substitution under selection."""
    neutral_host = 0
    selected_host = 0
    neutral_pathogen = 0
    selected_pathogen = 0
    c, v = 0.5, 0.5  # Initial traits
    ch, cp, time_now = 0, 0, 0.0
    max_f_H, max_f_P = -999, -999
    results = []

    # Use Gaussian-weighted bins
    step_bins_move, step_probs_move = gaussian_step_bins(n_steps=51, std_dev=stdDevMove)

    while time_now < max_evo_time:
        muts, rates = [], []

        # Host mutations
        for _ in range(int(num_mutations * gamma)):
            delta = np.random.choice(step_bins_move, p=step_probs_move)
            c_new = c + delta

            f_old = f_H(c, v)
            f_new = f_H(c_new, v)
            if not np.isfinite(f_old) or not np.isfinite(f_new):
                continue
            s = (f_new - f_old) / f_old

            if abs(s) < 1e-8:
                neutral_host += 1
            else:
                selected_host += 1

            rates.append(fixation_prob(s, Ne_H))
            muts.append(('host', delta))

        # Pathogen mutations
        for _ in range(int(num_mutations * (1 - gamma))):
            delta = np.random.choice(step_bins_move, p=step_probs_move)
            v_new = v + delta

            f_old = f_P(c, v)
            f_new = f_P(c, v_new)
            if not np.isfinite(f_old) or not np.isfinite(f_new):
                continue
            s = (f_new - f_old) / f_old

            if abs(s) < 1e-8:
                neutral_pathogen += 1
            else:
                selected_pathogen += 1

            rates.append(fixation_prob(s, Ne_P))
            muts.append(('pathogen', delta))

        total_rate = np.sum(rates)
        if total_rate == 0 or not np.isfinite(total_rate):
            time_now += 1e-3  # Small step forward to avoid freezing
            continue

        dwell_time = min(np.random.exponential(1 / total_rate), max_dwell_time)
        time_now += dwell_time

        kind, delta = muts[np.random.choice(len(muts), p=np.array(rates) / total_rate)]
        if kind == 'host':
            c += delta
            ch += 1
        else:
            v += delta
            cp += 1

        f_H_val, f_P_val = f_H(c, v), f_P(c, v)
        max_f_H = max(max_f_H, f_H_val)
        max_f_P = max(max_f_P, f_P_val)

        if len(results) % 100 == 0:
            print(f"[DEBUG] t={time_now:.2f}, c={c:.3f}, v={v:.3f}, f_H={f_H_val:.2e}, f_P={f_P_val:.2e}")

        results.append({
            "time": time_now, "c": c, "v": v,
            "f_H": f_H_val, "f_P": f_P_val,
            "omega_H": (ch / (time_now + 1e-6)) / (1 / Ne_H),
            "omega_P": (cp / (time_now + 1e-6)) / (1 / Ne_P),
            "dwell_time": dwell_time,
            "max_f_H": max_f_H, "max_f_P": max_f_P,
            "theta_c": np.nan, "theta_v": np.nan
        })

    return pd.DataFrame(results), neutral_host, selected_host, neutral_pathogen, selected_pathogen


# === ES SIMULATIONS ===

def simulate_ES(f_H, f_P):
    c_0, theta_c = 0.5, 0.0
    v_0, theta_v = 0.5, 0.0
    res, ch, cp, time_now = [], 0, 0, 0.0

    neutral_host, selected_host = 0, 0
    neutral_pathogen, selected_pathogen = 0, 0

    def find_equilibrium_ES(c0, t_c, v0, t_v):
        try:
            tan_c, tan_v = np.tan(t_c), np.tan(t_v)
        except:
            return None

        if abs(tan_c) > 10 or abs(tan_v) > 10:
            return None

        m_s, b_s = tan_c, c0 - tan_c * v0
        m_v, b_v = tan_v, v0 - tan_v * c0

        denom = 1 - m_s * m_v
        if abs(denom) < 1e-10:
            return None

        v = (b_s + m_s * b_v) / denom
        c = b_v + m_v * v

        if not (0.05 <= c <= 0.95 and 0.05 <= v <= 0.95):
            return None

        return (c, v)

    while time_now < max_evo_time:
        muts, rates = [], []

        # --- Host mutations
        for _ in range(int(num_mutations * gamma)):
            param = np.random.choice([0, 1])
            delta = np.random.choice(move_step if param == 0 else angle_step,
                                     p=move_prob if param == 0 else angle_prob)
            c0n, tcn = c_0, theta_c
            if param == 0: c0n += delta
            else: tcn += delta

            pt_old = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
            pt_new = find_equilibrium_ES(c0n, tcn, v_0, theta_v)
            if pt_old is None or pt_new is None:
                continue

            f_old = f_H(*pt_old)
            f_new = f_H(*pt_new)
            if not np.isfinite(f_old) or not np.isfinite(f_new):
                continue

            s = (f_new - f_old) / f_old
            rates.append(fixation_prob(s, Ne_H))
            muts.append(('host', param, delta))

            if abs(s) < 1e-8:
                neutral_host += 1
            else:
                selected_host += 1

        # --- Pathogen mutations
        for _ in range(int(num_mutations * (1 - gamma))):
            param = np.random.choice([0, 1])
            delta = np.random.choice(move_step if param == 0 else angle_step,
                                     p=move_prob if param == 0 else angle_prob)
            v0n, tvn = v_0, theta_v
            if param == 0: v0n += delta
            else: tvn += delta

            pt_old = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
            pt_new = find_equilibrium_ES(c_0, theta_c, v0n, tvn)
            if pt_old is None or pt_new is None:
                continue

            f_old = f_P(*pt_old)
            f_new = f_P(*pt_new)
            if not np.isfinite(f_old) or not np.isfinite(f_new):
                continue

            s = (f_new - f_old) / f_old
            rates.append(fixation_prob(s, Ne_P))
            muts.append(('pathogen', param, delta))

            if abs(s) < 1e-8:
                neutral_pathogen += 1
            else:
                selected_pathogen += 1

        # === Advance simulation
        total_rate = np.sum(rates)
        if total_rate == 0 or not np.isfinite(total_rate):
            time_now += 1e-3
            continue

        dwell_time = np.random.exponential(1 / total_rate)
        time_now += dwell_time

        # === Apply fixed mutation
        kind, param, delta = muts[np.random.choice(len(muts), p=np.array(rates) / total_rate)]
        if kind == 'host':
            if param == 0:
                c_0 += delta
            else:
                theta_c = np.clip(theta_c + delta, -1.4, 1.4)
            ch += 1
        else:
            if param == 0:
                v_0 += delta
            else:
                theta_v = np.clip(theta_v + delta, -1.4, 1.4)
            cp += 1

        pt = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
        if pt is None:
            continue

        c, v = pt
        f_H_val, f_P_val = f_H(c, v), f_P(c, v)

        res.append({
            "time": time_now, "c": c, "v": v,
            "f_H": f_H_val, "f_P": f_P_val,
            "dwell_time": dwell_time,
            "omega_H": (ch / (time_now + 1e-6)) / (1 / Ne_H),
            "omega_P": (cp / (time_now + 1e-6)) / (1 / Ne_P),
            "theta_c": theta_c, "theta_v": theta_v,
            "c_0": c_0, "v_0": v_0
        })

        if len(res) % 100 == 0:
            print(f"[DEBUG] t={time_now:.2f}, c={c:.3f}, v={v:.3f}, f_H={f_H_val:.2e}, f_P={f_P_val:.2e}")

    return pd.DataFrame(res), neutral_host, selected_host, neutral_pathogen, selected_pathogen


# === MAIN EXECUTION ===
if __name__ == "__main__":
    import os
    import time

    start = time.time()

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # === EI: Evolutionary Interaction ===
    print("Running EI - Acute")
    df_ei_acute, nh_ei_a, sh_ei_a, np_ei_a, sp_ei_a = simulate_EI(f_H_acute, f_P_acute)
    df_ei_acute.to_csv("results/ei_acute.csv", index=False)
    print(f"EI-Acute: host ({nh_ei_a} neutral, {sh_ei_a} selected), pathogen ({np_ei_a} neutral, {sp_ei_a} selected)")

    print("Running EI - Chronic")
    df_ei_chronic, nh_ei_c, sh_ei_c, np_ei_c, sp_ei_c = simulate_EI(f_H_chronic, f_P_chronic)
    df_ei_chronic.to_csv("results/ei_chronic.csv", index=False)
    print(f"EI-Chronic: host ({nh_ei_c} neutral, {sh_ei_c} selected), pathogen ({np_ei_c} neutral, {sp_ei_c} selected)")

    # === ES: Evolutionary Strategy ===
    print("Running ES - Acute")
    df_es_acute, nh_es_a, sh_es_a, np_es_a, sp_es_a = simulate_ES(f_H_acute, f_P_acute)
    df_es_acute.to_csv("results/es_acute.csv", index=False)
    print(f"ES-Acute: host ({nh_es_a} neutral, {sh_es_a} selected), pathogen ({np_es_a} neutral, {sp_es_a} selected)")

    print("Running ES - Chronic")
    df_es_chronic, nh_es_c, sh_es_c, np_es_c, sp_es_c = simulate_ES(f_H_chronic, f_P_chronic)
    df_es_chronic.to_csv("results/es_chronic.csv", index=False)
    print(f"ES-Chronic: host ({nh_es_c} neutral, {sh_es_c} selected), pathogen ({np_es_c} neutral, {sp_es_c} selected)")

    print(f"\nAll done in {round(time.time() - start, 2)} seconds.")