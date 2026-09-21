import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate

os.makedirs("figures", exist_ok=True)

def solve_deterministic_sir(R0, N, initial_infectives) -> float:
    """Solves SIR ODEs to find the final epidemic size (S_inf/N)."""
    gamma = 1.0
    beta = R0 * gamma
    
    def sir_model(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    y0 = [N - initial_infectives, initial_infectives, 0]
    t_span = (0, 1000) 
    sol = integrate.solve_ivp(sir_model, t_span, y0, method='RK45', 
                              events=lambda t, y: y[1] - 1e-3, 
                              abs_tol=1e-6, rel_tol=1e-6)
    
    s_inf = sol.y[0, -1]
    return (N - s_inf) / N

def run_gillespie_sir(R0, N, initial_infectives, rng) -> float:
    """Runs a single stochastic SIR simulation using the Gillespie method."""
    gamma = 1.0
    beta = R0 * gamma
    
    S = N - initial_infectives
    I = initial_infectives
    R = 0
    
    while I > 0:
        rate_inf = beta * S * I / N
        rate_rec = gamma * I
        total_rate = rate_inf + rate_rec
        
        if total_rate == 0:
            break
            
        if rng.random() < rate_inf / total_rate:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
            
    return (N - S) / N

def analyze_outbreaks(final_sizes, N, threshold_fraction) -> tuple:
    """Calculates outbreak probability and mean size of major outbreaks."""
    final_sizes = np.array(final_sizes)
    major_mask = final_sizes > threshold_fraction
    prob = np.mean(major_mask)
    
    if prob > 0:
        mean_major = np.mean(final_sizes[major_mask])
    else:
        mean_major = 0.0
        
    return float(prob), float(mean_major)

def plot_outbreak_prob(results) -> None:
    R0_vals = sorted(results.keys())
    N_vals = sorted(list(results[R0_vals[0]].keys()))
    plt.figure(figsize=(6, 4))
    r0_fine = np.linspace(1.01, max(R0_vals), 100)
    plt.plot(r0_fine, 1 - 1/r0_fine, color='black', linestyle='--', label='Theoretical (1-1/R0)')
    for n in N_vals:
        probs = [results[r0][n]['outbreak_probability'] for r0 in R0_vals]
        plt.plot(R0_vals, probs, marker='o', label=f'N={n}')
    plt.xlabel("Basic Reproduction Number $R_0$")
    plt.ylabel("Outbreak Probability")
    plt.title("Probability of Major Outbreak vs $R_0$")
    plt.legend(frameon=False)
    plt.ylim(-0.05, 1.05)
    plt.savefig("figures/outbreak_prob_vs_R0.png", bbox_inches="tight")
    plt.close()

def plot_final_size_dist(results, R0_val, N_vals) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, n in enumerate(N_vals):
        ax = axes[i]
        data = results[R0_val][n]['all_final_sizes']
        sns.histplot(data, bins=30, ax=ax, color='steelblue')
        ax.set_title(f"N = {n}")
        ax.set_xlabel("Final Size ($S_{\infty}/N $")
        if i == 0:
            ax.set_ylabel("Frequency")
    fig.suptitle(f"Final Size Distribution for $R_0 = {R0_val}$", y=1.02)
    fig.tight_layout()
    plt.savefig("figures/final_size_dist_N_sweep.png", bbox_inches="tight")
    plt.close()

def plot_convergence(results) -> None:
    R0_vals = sorted(results.keys())
    N_vals = sorted(list(results[R0_vals[0]].keys()))
    plt.figure(figsize=(6, 4))
    for r0 in R0_vals:
        stoch_means = [results[r0][n]['mean_final_size_major'] for n in N_vals]
        det_means = [results[r0][n]['deterministic_final_size'] for n in N_vals]
        line, = plt.plot(N_vals, stoch_means, marker='o', label=f'$R_0={r0}$ (Stoch)')
        plt.plot(N_vals, det_means, color=line.get_color(), linestyle='--', alpha=0.6, label=f'$R_0={r0}$ (Det)')
    plt.xscale('log')
    plt.xlabel("Population Size N")
    plt.ylabel("Final Size Fraction")
    plt.title("Convergence of Major Outbreaks to Deterministic Limit")
    plt.legend(frameon=False, ncol=2)
    plt.savefig("figures/stochastic_vs_deterministic_convergence.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    seed = int(os.environ.get("FI_REPLICATE_SEED", 0))
    rng = np.random.default_rng(seed)
    
    R0_VALUES = [0.9, 1.5, 3.0]
    N_VALUES = [100, 1000, 5000]
    INITIAL_INFECTIVES = 1
    NUM_RUNS = 300
    MAJOR_THRESHOLD = 0.05

    if os.environ.get("FI_PILOT") == "1":
        NUM_RUNS = 30
        N_VALUES = [100, 500, 1000]

    all_results = {}
    for r0 in R0_VALUES:
        all_results[r0] = {}
        for n in N_VALUES:
            det_size = solve_deterministic_sir(r0, n, INITIAL_INFECTIVES)
            stoch_sizes = []
            for i in range(NUM_RUNS):
                stoch_sizes.append(run_gillespie_sir(r0, n, INITIAL_INFECTIVES, rng))
            
            prob, mean_major = analyze_outbreaks(stoch_sizes, n, MAJOR_THRESHOLD)
            all_results[r0][n] = {
                "deterministic_final_size": det_size,
                "outbreak_probability": prob,
                "mean_final_size_major": mean_major,
                "all_final_sizes": stoch_sizes
            }

    plot_outbreak_prob(all_results)
    current_n_vals = N_VALUES if os.environ.get("FI_PILOT") != "1" else [100, 500, 1000]
    plot_final_size_dist(all_results, 1.5, current_n_vals)
    plot_convergence(all_results)

    final_json = {}
    for r0 in all_results:
        for n in all_results[r0]:
            key = f"R0_{r0}_N_{n}"
            res = all_results[r0][n]
            final_json[key] = {
                "outbreak_probability": res["outbreak_probability"],
                "mean_final_size_major": res["mean_final_size_major"],
                "deterministic_final_size": res["deterministic_final_size"]
            }
    print(f"RESULT_JSON: {json.dumps(final_json)}")
