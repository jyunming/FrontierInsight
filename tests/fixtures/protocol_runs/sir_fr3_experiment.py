import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate

os.makedirs("figures", exist_ok=True)

def solve_deterministic_sir(R0, N) -> float:
    """Computes the final size of the epidemic using the deterministic SIR ODE."""
    # SIR ODEs: dS/dt = -beta*S*I/N, dI/dt = beta*S*I/N - gamma*I, dR/dt = gamma*I
    # R0 = beta / gamma. Let gamma = 1 for time scaling.
    beta = R0
    gamma = 1.0
    
    def sir_derivs(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    # Initial conditions: 1 infective, N-1 susceptible
    y0 = [N - 1, 1, 0]
    # Integrate until I is effectively 0. 
    # For R0 > 1, the epidemic lasts a finite time.
    t_span = (0, 1000) 
    sol = integrate.solve_ivp(sir_derivs, t_span, y0, method='RK45', 
                              rtol=1e-6, atol=1e-6, 
                              events=lambda t, y: y[1] - 1e-4)
    
    # Final size is R(infinity) / N
    return sol.y[2, -1] / N

def simulate_gillespie_sir(R0, N, initial_infectives) -> int:
    """Simulates a single stochastic SIR process and returns the final number of recovered individuals."""
    # Rates: 
    # Infection: beta * S * I / N
    # Recovery: gamma * I
    # Let gamma = 1.0, then beta = R0.
    beta = R0
    gamma = 1.0
    
    S = N - initial_infectives
    I = initial_infectives
    R = 0
    
    # Use a local RNG for reproducibility based on seed
    seed_val = int(os.environ.get("FI_REPLICATE_SEED", 0))
    rng = np.random.default_rng(seed_val)
    
    while I > 0:
        rate_inf = beta * S * I / N
        rate_rec = gamma * I
        total_rate = rate_inf + rate_rec
        
        if total_rate == 0:
            break
            
        # Time to next event
        # dt = rng.exponential(1/total_rate) # Not needed for final size
        
        # Which event occurs?
        if rng.random() < rate_inf / total_rate:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
            
    return R

def run_stochastic_experiment(R0, N, initial_infectives, num_runs, threshold_fraction) -> dict:
    """Runs multiple Gillespie simulations and computes outbreak probability and conditional statistics."""
    seed_val = int(os.environ.get("FI_REPLICATE_SEED", 0))
    rng = np.random.default_rng(seed_val)
    
    final_sizes = []
    # To ensure each run is different but reproducible, we derive seeds
    for i in range(num_runs):
        # We pass a modified seed to the simulation logic if it used one, 
        # but here we just use the rng object for the whole experiment.
        # Re-implementing the loop inside simulate_gillespie_sir using the passed rng:
        S, I, R = N - initial_infectives, initial_infectives, 0
        beta, gamma = R0, 1.0
        while I > 0:
            r_inf = beta * S * I / N
            r_rec = gamma * I
            tot = r_inf + r_rec
            if tot == 0: break
            if rng.random() < r_inf / tot:
                S -= 1
                I += 1
            else:
                I -= 1
                R += 1
        final_sizes.append(R)
        
    final_sizes = np.array(final_sizes)
    major_mask = final_sizes > (threshold_fraction * N)
    prob = np.mean(major_mask)
    
    if prob > 0:
        major_outbreaks = final_sizes[major_mask] / N
        mean_val = np.mean(major_outbreaks)
        var_val = np.var(major_outbreaks)
    else:
        mean_val = 0.0
        var_val = 0.0
        
    return {"prob": float(prob), "mean": float(mean_val), "var": float(var_val), "raw": final_sizes}

def plot_outbreak_prob(R0_vals, N_vals, probs, theoretical_probs) -> None:
    """Saves outbreak_prob_vs_N.png"""
    fig, ax = plt.subplots(1, 1)
    for r0, r0_idx in zip(R0_vals, range(len(R0_vals))):
        p_vals = [probs[r0_idx][n_idx] for n_idx in range(len(N_vals))]
        ax.plot(N_vals, p_vals, marker='o', label=f"R0={r0}")
        
        # Theoretical 1 - 1/R0 (only for R0 > 1)
        if r0 > 1:
            theo = 1 - 1/r0
            ax.axhline(theo, color='gray', linestyle='--', alpha=0.5)
            
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Outbreak Probability")
    ax.set_title("Outbreak Probability vs Population Size")
    ax.legend(frameon=False)
    plt.savefig("figures/outbreak_prob_vs_N.png")
    plt.close()

def plot_final_size_convergence(R0_vals, N_vals, conditional_means, deterministic_sizes) -> None:
    """Saves final_size_convergence.png"""
    fig, ax = plt.subplots(1, 1)
    for r0, r0_idx in zip(R0_vals, range(len(R0_vals))):
        means = [conditional_means[r0_idx][n_idx] for n_idx in range(len(N_vals))]
        dets = [deterministic_sizes[r0_idx][n_idx] for n_idx in range(len(N_vals))]
        ax.plot(N_vals, means, marker='o', label=f"Stoch R0={r0}")
        ax.plot(N_vals, dets, marker='x', linestyle='--', label=f"Det R0={r0}")
        
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Final Size (Relative)")
    ax.set_title("Convergence of Conditional Final Size to Deterministic Limit")
    ax.legend(frameon=False, ncol=2)
    plt.savefig("figures/final_size_convergence.png")
    plt.close()

def plot_stochastic_distribution(R0, N, final_sizes, deterministic_size) -> None:
    """Saves stochastic_distribution_R0_3.png"""
    fig, ax = plt.subplots(1, 1)
    # Normalize final sizes to relative
    rel_sizes = final_sizes / N
    ax.hist(rel_sizes, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(deterministic_size, color='red', linewidth=2, label=f"Deterministic: {deterministic_size:.3f}")
    ax.set_xlabel("Final Epidemic Size (Relative)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Final Sizes (R0={R0}, N={N})")
    ax.legend(frameon=False)
    plt.savefig("figures/stochastic_distribution_R0_3.png")
    plt.close()

if __name__ == "__main__":
    # Constants
    R0_LIST = [0.8, 1.2, 2.0, 3.0]
    N_LIST = [100, 500, 1000, 5000]
    INITIAL_INFECTIVES = 1
    NUM_RUNS = 300
    THRESHOLD_FRACTION = 0.05

    # FI_PILOT handling
    if os.environ.get("FI_PILOT") == "1":
        N_LIST = [100, 500]
        NUM_RUNS = 50

    results_by_R0 = {}
    
    # For plotting
    all_probs = []
    all_means = []
    all_dets = []
    
    # Store raw data for the specific distribution plot (R0=3.0, N=max)
    dist_data = None
    dist_det = 0.0

    for r0 in R0_LIST:
        r0_str = str(r0)
        results_by_R0[r0_str] = {}
        
        r0_probs = []
        r0_means = []
        r0_dets = []
        
        for n in N_LIST:
            det_size = solve_deterministic_sir(r0, n)
            stoch_stats = run_stochastic_experiment(r0, n, INITIAL_INFECTIVES, NUM_RUNS, THRESHOLD_FRACTION)
            
            results_by_R0[r0_str][str(n)] = {
                "outbreak_probability": stoch_stats["prob"],
                "mean_final_size_conditional_relative": stoch_stats["mean"],
                "final_size_variance_relative": stoch_stats["var"],
                "deterministic_final_size_relative": det_size
            }
            
            r0_probs.append(stoch_stats["prob"])
            r0_means.append(stoch_stats["mean"])
            r0_dets.append(det_size)
            
            if r0 == 3.0 and n == N_LIST[-1]:
                dist_data = stoch_stats["raw"]
                dist_det = det_size
                
        all_probs.append(r0_probs)
        all_means.append(r0_means)
        all_dets.append(r0_dets)

    # Figure generation
    plot_outbreak_prob(R0_LIST, N_LIST, all_probs, [1-1/r if r>1 else 0 for r in R0_LIST])
    plot_final_size_convergence(R0_LIST, N_LIST, all_means, all_dets)
    if dist_data is not None:
        plot_stochastic_distribution(3.0, N_LIST[-1], dist_data, dist_det)

    print(f"RESULT_JSON: {json.dumps({'by_R0': results_by_R0})}")