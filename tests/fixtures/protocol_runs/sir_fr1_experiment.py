import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
import pandas as pd

os.makedirs("figures", exist_ok=True)

def solve_deterministic_sir(R0, N, initial_infected) -> float:
    """Solves SIR ODE to find the final size relative to N."""
    # SIR ODEs:
    # dS/dt = -beta * S * I / N
    # dI/dt = beta * S * I / N - gamma * I
    # dR/dt = gamma * I
    # R0 = beta / gamma. Let gamma = 1 (time unit is 1/gamma).
    gamma = 1.0
    beta = R0 * gamma
    
    def sir_model(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    y0 = [N - initial_infected, initial_infected, 0]
    # Integrate until I is effectively zero. 
    # For R0 > 1, the epidemic lasts roughly proportional to N.
    # A safe upper bound for t is needed.
    t_span = (0, 1000) 
    
    # Use a dense t_eval or just the final point.
    sol = integrate.solve_ivp(sir_model, t_span, y0, method='RK45', 
                              rtol=1e-6, atol=1e-6)
    
    # Final size is R_inf / N
    return sol.y[2, -1] / N

def run_gillespie_sir(R0, N, initial_infected) -> float:
    """Runs a single stochastic SIR simulation using Gillespie Direct Method."""
    gamma = 1.0
    beta = R0 * gamma
    
    S = N - initial_infected
    I = initial_infected
    R = 0
    
    # Seed handled by global np.random
    while I > 0:
        # Rates
        rate_inf = beta * S * I / N
        rate_rec = gamma * I
        total_rate = rate_inf + rate_rec
        
        if total_rate == 0:
            break
            
        # Time to next event
        # t += np.random.exponential(1/total_rate) # Time not needed for final size
        
        # Which event?
        if np.random.random() < rate_inf / total_rate:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
            
    return R / N

def analyze_outbreaks(final_sizes, N, threshold_fraction=0.05) -> dict:
    """Calculates outbreak probability and mean size of major outbreaks."""
    final_sizes = np.array(final_sizes)
    major_mask = final_sizes > threshold_fraction
    prob = np.mean(major_mask)
    
    if prob > 0:
        mean_major = np.mean(final_sizes[major_mask])
    else:
        mean_major = 0.0
        
    return {"prob": float(prob), "mean_major": float(mean_major)}

def plot_distributions(results_df, filename) -> None:
    """Plots final size distributions faceted by R0 and N."""
    # We need the raw sizes. results_df contains 'stoch_sizes' as lists.
    # Expand the dataframe to long format for seaborn
    expanded_data = []
    for _, row in results_df.iterrows():
        for size in row['stoch_sizes']:
            expanded_data.append({'R0': row['R0'], 'N': row['N'], 'Size': size})
    
    df_long = pd.DataFrame(expanded_data)
    
    # To keep panels <= 3, we plot R0s separately or Ns separately.
    # We'll create one figure per R0, with 3 panels for N.
    for r0 in results_df['R0'].unique():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        subset = df_long[df_long['R0'] == r0]
        
        for i, n in enumerate(results_df['N'].unique()):
            ax = axes[i]
            data = subset[subset['N'] == n]['Size']
            sns.histplot(data, bins=30, ax=ax, stat="density", color="steelblue")
            ax.set_title(f"N={n}")
            ax.set_xlabel("Final Size (relative)")
            if i == 0:
                ax.set_ylabel("Density")
        
        fig.suptitle(f"Final Size Distribution for R0 = {r0}")
        plt.tight_layout()
        plt.savefig(f"figures/{filename.replace('.png', f'_R0_{r0}.png')}")
        plt.close()

def plot_probability_convergence(results_df, filename) -> None:
    """Plots outbreak probability vs N for different R0."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    for r0 in results_df['R0'].unique():
        subset = results_df[results_df['R0'] == r0]
        # Theoretical limit: 1 - 1/R0 (for R0 > 1)
        theory = max(0, 1 - 1/r0)
        
        ax.plot(subset['N'], subset['outbreak_probability'], marker='o', label=f"R0={r0}")
        ax.axhline(theory, color='gray', linestyle='--', alpha=0.5, label=f"Limit R0={r0}" if r0 > 1 else None)
        
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Outbreak Probability")
    ax.set_title("Convergence of Outbreak Probability")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()

def plot_size_comparison(results_df, filename) -> None:
    """Plots deterministic vs stochastic major outbreak sizes."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    for r0 in results_df['R0'].unique():
        subset = results_df[results_df['R0'] == r0]
        ax.plot(subset['N'], subset['deterministic_final_size_relative'], marker='s', label=f"Det R0={r0}")
        ax.plot(subset['N'], subset['mean_final_size_major_relative'], marker='o', label=f"Stoch Major R0={r0}")
        
    ax.set_xlabel("Population Size (N)")
    ax.set_ylabel("Relative Final Size")
    ax.set_title("Deterministic vs Stochastic Major Outbreak Size")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close()

if __name__ == "__main__":
    # Handle Seed
    seed_env = os.environ.get("FI_REPLICATE_SEED", "0")
    seed = int(seed_env)
    np.random.seed(seed)
    
    # Handle Pilot
    is_pilot = os.environ.get("FI_PILOT", "0") == "1"
    
    # Constants
    R0_VALUES = [0.9, 1.5, 3.0]
    N_VALUES = [100, 1000, 5000]
    INITIAL_INFECTED = 1
    NUM_SIMULATIONS = 300 if not is_pilot else 50
    
    all_data = []
    
    for r0 in R0_VALUES:
        for n in N_VALUES:
            # 1. Deterministic
            det_size = solve_deterministic_sir(r0, n, INITIAL_INFECTED)
            
            # 2. Stochastic
            stoch_sizes = [run_gillespie_sir(r0, n, INITIAL_INFECTED) for _ in range(NUM_SIMULATIONS)]
            
            # 3. Analysis
            stats = analyze_outbreaks(stoch_sizes, n)
            
            all_data.append({
                "R0": r0,
                "N": n,
                "deterministic_final_size_relative": det_size,
                "outbreak_probability": stats["prob"],
                "mean_final_size_major_relative": stats["mean_major"],
                "stoch_sizes": stoch_sizes
            })

    results_df = pd.DataFrame(all_data)
    
    # Figures
    plot_distributions(results_df, "final_size_distributions_faceted_by_R0_N.png")
    plot_probability_convergence(results_df, "outbreak_probability_convergence_vs_N.png")
    plot_size_comparison(results_df, "deterministic_vs_stochastic_size_comparison.png")

    # Final Result JSON
    result_json = {"by_R0": {}}
    for r0 in R0_VALUES:
        r0_str = str(r0)
        result_json["by_R0"][r0_str] = {"by_N": {}}
        for n in N_VALUES:
            n_str = str(n)
            row = results_df[(results_df['R0'] == r0) & (results_df['N'] == n)].iloc[0]
            result_json["by_R0"][r0_str]["by_N"][n_str] = {
                "outbreak_probability": row["outbreak_probability"],
                "mean_final_size_major_relative": row["mean_final_size_major_relative"],
                "deterministic_final_size_relative": row["deterministic_final_size_relative"]
            }
            
    print(f"RESULT_JSON: {json.dumps(result_json)}")