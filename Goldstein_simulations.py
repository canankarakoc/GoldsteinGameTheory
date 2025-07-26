import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# === MODEL PARAMETERS ===
beta = 1
alpha = 2
d_0 = 0.1
m_c = 0.1
m_v = 1
epsilon = 1e-4
mutation_step_c_v = 0.1
mutation_step_theta = 0.314

gamma = 0.01
Ne_H = 10_000
Ne_P = 1_000_000
num_mutations = 10000
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
    if not np.isfinite(s) or abs(s) < 1e-8:
        return 1 / Ne
    try:
        exp_num = np.exp(-2 * s)
        exp_denom = np.exp(-4 * Ne * s) if -4 * Ne * s > -700 else 0.0
        return (1 - exp_num) / (1 - exp_denom) if exp_denom != 0 else 1 / Ne
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

# === SIMULATORS ===
def simulate_EI(f_H, f_P):
    c, v = 0.5, 0.5
    res, ch, cp, time_now = [], 0, 0, 0.0

    while time_now < max_evo_time:
        muts, rates = [], []

        # --- Host mutations
        for _ in range(int(num_mutations * gamma)):
            delta = np.random.normal(0, mutation_step_c_v)
            c_new = np.clip(c + delta, 0, 1)
            s = (f_H(c_new, v) - f_H(c, v)) / f_H(c, v)
            p_fix = fixation_prob(s, Ne_H)
            muts.append(('host', delta))
            rates.append(p_fix)

        # --- Pathogen mutations
        for _ in range(int(num_mutations * (1 - gamma))):
            delta = np.random.normal(0, mutation_step_c_v)
            v_new = np.clip(v + delta, 0, 1)
            s = (f_P(c, v_new) - f_P(c, v)) / f_P(c, v)
            p_fix = fixation_prob(s, Ne_P)
            muts.append(('pathogen', delta))
            rates.append(p_fix)

        # --- Calculate total substitution rate
        total_rate = np.sum(rates)
        if total_rate == 0 or np.isnan(total_rate):
            # skip this round to avoid division error
            continue

        # --- Sample dwell time from exponential
        dwell_time = np.random.exponential(1 / total_rate)
        time_now += dwell_time

        # --- Choose mutation to fix
        probs = np.array(rates) / total_rate
        idx = np.random.choice(len(muts), p=probs)
        kind, delta = muts[idx]

        # --- Apply substitution
        if kind == 'host':
            c = np.clip(c + delta, 0, 1)
            ch += 1
        else:
            v = np.clip(v + delta, 0, 1)
            cp += 1

        # --- Record results
        res.append({
            "time": time_now, "c": c, "v": v,
            "f_H": f_H(c, v), "f_P": f_P(c, v),
            "dwell_time": dwell_time,
            "omega_H": (ch / (time_now + 1e-6)) / (1 / Ne_H),
            "omega_P": (cp / (time_now + 1e-6)) / (1 / Ne_P)
        })

    return res

def simulate_ES(f_H, f_P):
    c_0, theta_c = 0.5, 0.0
    v_0, theta_v = 0.5, 0.0
    res, ch, cp, time_now = [], 0, 0, 0.0

    while time_now < max_evo_time:
        muts, rates = [], []

        # --- Host mutations: c_0 or theta_c
        for _ in range(int(num_mutations * gamma)):
            param = np.random.choice([0, 1])
            delta = np.random.normal(0, mutation_step_theta if param else mutation_step_c_v)
            c0n, tcn = c_0, theta_c
            if param == 0: c0n += delta
            else: tcn += delta
            c_new, v_new = find_equilibrium_ES(c0n, tcn, v_0, theta_v)
            c_old, v_old = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
            s = (f_H(c_new, v_new) - f_H(c_old, v_old)) / f_H(c_old, v_old)
            p_fix = fixation_prob(s, Ne_H)
            muts.append(('host', param, delta))
            rates.append(p_fix)

        # --- Pathogen mutations: v_0 or theta_v
        for _ in range(int(num_mutations * (1 - gamma))):
            param = np.random.choice([0, 1])
            delta = np.random.normal(0, mutation_step_theta if param else mutation_step_c_v)
            v0n, tvn = v_0, theta_v
            if param == 0: v0n += delta
            else: tvn += delta
            c_new, v_new = find_equilibrium_ES(c_0, theta_c, v0n, tvn)
            c_old, v_old = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)
            s = (f_P(c_new, v_new) - f_P(c_old, v_old)) / f_P(c_old, v_old)
            p_fix = fixation_prob(s, Ne_P)
            muts.append(('pathogen', param, delta))
            rates.append(p_fix)

        # --- Compute total substitution rate
        total_rate = np.sum(rates)
        if total_rate == 0 or np.isnan(total_rate):
            continue

        # --- Sample dwell time
        dwell_time = np.random.exponential(1 / total_rate)
        time_now += dwell_time

        # --- Pick substitution to fix
        probs = np.array(rates) / total_rate
        idx = np.random.choice(len(muts), p=probs)
        kind, param, delta = muts[idx]

        # --- Apply mutation
        if kind == 'host':
            if param == 0: c_0 += delta
            else: theta_c += delta
            ch += 1
        else:
            if param == 0: v_0 += delta
            else: theta_v += delta
            cp += 1

        # --- Calculate new strategy equilibrium
        c, v = find_equilibrium_ES(c_0, theta_c, v_0, theta_v)

        # --- Record outcome
        res.append({
            "time": time_now, "c": c, "v": v,
            "f_H": f_H(c, v), "f_P": f_P(c, v),
            "dwell_time": dwell_time,
            "omega_H": (ch / (time_now + 1e-6)) / (1 / Ne_H),
            "omega_P": (cp / (time_now + 1e-6)) / (1 / Ne_P),
            "theta_c": theta_c, "theta_v": theta_v
        })

    return res

# === WRAPPER ===
def run_simulation_replicate(sim_type="EI", infection_type="acute", seed=None):
    np.random.seed(seed)
    f_H, f_P = {
        "acute": (f_H_acute, f_P_acute),
        "chronic": (f_H_chronic, f_P_chronic)
    }[infection_type]

    sim_fn = simulate_EI if sim_type == "EI" else simulate_ES
    data = sim_fn(f_H, f_P)
    for row in data:
        row["model"] = sim_type
        row["scenario"] = infection_type
        row["replicate"] = seed
    return pd.DataFrame(data)

# === MAIN RUN ===
if __name__ == "__main__":
    start = time.time()
    all_dfs = []
    for model in ["EI", "ES"]:
        for scenario in ["acute", "chronic"]:
            for rep in range(10):
                print(f"Running {model}-{scenario}, Replicate {rep}")
                df = run_simulation_replicate(model, scenario, rep)
                all_dfs.append(df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv("./results/all_simulations_replicates.csv", index=False)
    print("[DONE] Simulations completed.")
    print(f"Total time: {time.time() - start:.2f} seconds")