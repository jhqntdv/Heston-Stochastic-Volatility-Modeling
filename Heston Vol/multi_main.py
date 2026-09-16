import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heston import (
    HestonParameters,
    HestonSimulatorFT,
    HestonSimulatorQE,
    PriceCorrection,
)


def _asian_price(S_paths, K, r, T):
    """Compute discounted payoff of an Asian option with averaging over terminal steps."""
    n_avg = -1
    S_avg = np.mean(S_paths[:, n_avg:], axis=1)
    return np.maximum(S_avg - K, 0.0) * np.exp(-r * T)


def _worker_seed_task(args):
    """Worker task to simulate a single seed for a given scheme."""
    (
        name,
        sim_cls,
        corr,
        current_seed,
        base_params,
        S0,
        K,
        T,
        r,
        q,
        n_steps,
        n_sims,
        dS,
        dv,
        Df,
        parity_target,
        theoretical_fwd,
    ) = args

    # 1. Base simulation
    sim_base = sim_cls(base_params, seed=current_seed)
    _, S_base, _ = sim_base.simulate_paths(
        S0, T, r, n_steps=n_steps, n_sims=n_sims, q=q, price_correction=corr
    )

    payoffs = _asian_price(S_base, K, r, T)
    p_base = float(payoffs.mean())
    naive_se = float(payoffs.std(ddof=1) / np.sqrt(n_sims))

    simulated_fwd = np.mean(S_base, axis=0)
    drift_bias = simulated_fwd - theoretical_fwd

    # Put-Call Parity check
    S_T = S_base[:, -1]
    c_mc = float(np.mean(np.maximum(S_T - K, 0.0)) * Df)
    p_mc = float(np.mean(np.maximum(K - S_T, 0.0)) * Df)
    parity_error = (c_mc - p_mc) - parity_target

    # 2. Spot bumps for Delta & Gamma (Common Random Numbers)
    sim_up = sim_cls(base_params, seed=current_seed)
    _, S_up, _ = sim_up.simulate_paths(
        S0 + dS, T, r, n_steps=n_steps, n_sims=n_sims, q=q, price_correction=corr
    )
    p_up = float(_asian_price(S_up, K, r, T).mean())

    sim_dn = sim_cls(base_params, seed=current_seed)
    _, S_dn, _ = sim_dn.simulate_paths(
        S0 - dS, T, r, n_steps=n_steps, n_sims=n_sims, q=q, price_correction=corr
    )
    p_dn = float(_asian_price(S_dn, K, r, T).mean())

    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * p_base + p_dn) / (dS ** 2)

    # 3. Variance bumps for Vega (Common Random Numbers)
    p_vup = HestonParameters(
        kappa=base_params.kappa,
        theta=base_params.theta,
        xi=base_params.xi,
        rho=base_params.rho,
        v_0=base_params.v_0 + dv,
    )
    sim_vup = sim_cls(p_vup, seed=current_seed)
    _, S_vup, _ = sim_vup.simulate_paths(
        S0, T, r, n_steps=n_steps, n_sims=n_sims, q=q, price_correction=corr
    )
    pv_up = float(_asian_price(S_vup, K, r, T).mean())

    p_vdn = HestonParameters(
        kappa=base_params.kappa,
        theta=base_params.theta,
        xi=base_params.xi,
        rho=base_params.rho,
        v_0=base_params.v_0 - dv,
    )
    sim_vdn = sim_cls(p_vdn, seed=current_seed)
    _, S_vdn, _ = sim_vdn.simulate_paths(
        S0, T, r, n_steps=n_steps, n_sims=n_sims, q=q, price_correction=corr
    )
    pv_dn = float(_asian_price(S_vdn, K, r, T).mean())

    vega = (pv_up - pv_dn) / (2.0 * dv)

    return (
        name,
        current_seed,
        p_base,
        naive_se,
        drift_bias,
        parity_error,
        delta,
        gamma,
        vega,
    )


def main():
    print("=" * 80)
    print(" Heston Multiprocessing Simulation Benchmark")
    print("=" * 80)

    # Base parameters (Stress Test: xi=0.60, Feller Ratio=0.378)
    base_params = HestonParameters(
        kappa=1.70,
        theta=0.04,
        xi=0.60,
        rho=-0.70,
        v_0=0.09,
    )

    S0 = 120.0
    K = 120.0
    T = 1.0
    r = 0.02
    q = 0.0
    n_steps = 252

    # Simulation settings (can be overridden via CLI: python multi_main.py [n_seeds] [n_sims])
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    dv = 0.005
    start_seed = 42
    dS = S0 * 0.001

    Df = np.exp(-r * T)
    parity_target = S0 * np.exp(-q * T) - K * Df
    time_grid = np.linspace(0.0, T, n_steps + 1)
    theoretical_fwd = S0 * np.exp((r - q) * time_grid)

    # Determine CPU workers (Leave 2 threads for OS/UI responsiveness)
    cpu_cores = os.cpu_count() or 8
    max_workers = max(1, cpu_cores - 2)

    print(f"Detected CPU logical processors: {cpu_cores}")
    print(f"Using worker processes: {max_workers}")
    print(f"Parameters: kappa={base_params.kappa}, theta={base_params.theta}, xi={base_params.xi}, rho={base_params.rho}, v_0={base_params.v_0}")
    print(f"Feller Ratio: {base_params.feller_ratio:.4f} (Satisfied: {base_params.is_feller_satisfied})")
    print(f"Simulation setup: n_sims={n_sims:,}, n_seeds={n_seeds}, total seeds={n_seeds * 3}")
    print("-" * 80)

    schemes = [
        ("Heston FT (None)", HestonSimulatorFT, PriceCorrection.NONE),
        ("Heston QE (EMS)", HestonSimulatorQE, PriceCorrection.EMS),
        ("Heston QE (Andersen M-Corr)", HestonSimulatorQE, PriceCorrection.ANDERSEN),
    ]

    # Build task list
    tasks = []
    for name, sim_cls, corr in schemes:
        for i in range(n_seeds):
            current_seed = start_seed + i
            tasks.append(
                (
                    name,
                    sim_cls,
                    corr,
                    current_seed,
                    base_params,
                    S0,
                    K,
                    T,
                    r,
                    q,
                    n_steps,
                    n_sims,
                    dS,
                    dv,
                    Df,
                    parity_target,
                    theoretical_fwd,
                )
            )

    total_tasks = len(tasks)
    print(f"Total tasks to run: {total_tasks} across {len(schemes)} schemes.")
    print("Starting multiprocessing execution...")

    # Data structures for collecting results
    collected = {
        name: {
            "prices": [],
            "naive_ses": [],
            "seed_drift_biases": [],
            "seed_parity_errors": [],
            "deltas": [],
            "gammas": [],
            "vegas": [],
        }
        for name, _, _ in schemes
    }

    start_time = time.time()
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_seed_task, t): t for t in tasks}

        for future in as_completed(futures):
            res = future.result()
            (
                name,
                seed,
                p_base,
                naive_se,
                drift_bias,
                parity_error,
                delta,
                gamma,
                vega,
            ) = res

            collected[name]["prices"].append(p_base)
            collected[name]["naive_ses"].append(naive_se)
            collected[name]["seed_drift_biases"].append(drift_bias)
            collected[name]["seed_parity_errors"].append(parity_error)
            collected[name]["deltas"].append(delta)
            collected[name]["gammas"].append(gamma)
            collected[name]["vegas"].append(vega)

            completed_count += 1
            if completed_count % 15 == 0 or completed_count == total_tasks:
                elapsed = time.time() - start_time
                pct = completed_count / total_tasks * 100
                rate = completed_count / elapsed
                remaining = (total_tasks - completed_count) / rate if rate > 0 else 0
                print(
                    f"Progress: [{completed_count:3d}/{total_tasks}] "
                    f"({pct:5.1f}%) | Elapsed: {elapsed:5.1f}s | "
                    f"ETA: {remaining:5.1f}s",
                    flush=True,
                )

    total_time = time.time() - start_time
    print(f"\nAll tasks finished in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)!")
    print("=" * 90)

    # 1. Put-Call Parity Summary
    parity_results = {}
    for name, _, _ in schemes:
        errors = np.array(collected[name]["seed_parity_errors"])
        parity_mean = float(errors.mean())
        parity_se = float(errors.std(ddof=1) / np.sqrt(n_seeds))
        t_stat = parity_mean / parity_se if parity_se > 0 else np.nan
        parity_results[name] = {
            "mean": parity_mean,
            "se": parity_se,
            "t_stat": t_stat,
        }

    print("\n" + "-" * 30 + " Put-Call Parity Check (across seeds) " + "-" * 30)
    print(f"{'Scheme':<28} | {'Parity Error':<14} | {'SE':<10} | {'t-stat':<8}")
    print("-" * 90)
    for name, _, _ in schemes:
        res = parity_results[name]
        print(
            f"{name:<28} | {res['mean']:<14.6f} | {res['se']:<10.6f} | {res['t_stat']:<8.3f}"
        )
    print("-" * 90)

    # 2. Asian Call Option & Greeks Summary
    results = {}
    for name, _, _ in schemes:
        prices = collected[name]["prices"]
        naive_ses = collected[name]["naive_ses"]
        vegas = collected[name]["vegas"]
        vega_std = float(np.std(vegas, ddof=1)) if n_seeds > 1 else 0.0
        vega_se = float(vega_std / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
        results[name] = {
            "Price": float(np.mean(prices)),
            "True SE": float(np.std(prices, ddof=1) / np.sqrt(n_seeds))
            if n_seeds > 1
            else naive_ses[0],
            "Naive SE": float(np.mean(naive_ses)),
            "Delta": float(np.mean(collected[name]["deltas"])),
            "Gamma": float(np.mean(collected[name]["gammas"])),
            "Vega": float(np.mean(vegas)),
            "Vega Std": vega_std,
            "Vega SE": vega_se,
        }

    print("\n" + "-" * 54 + " Option Pricing & Greeks " + "-" * 54)
    print(
        f"{'Scheme':<28} | {'Price':<8} | {'Delta':<8} | {'Gamma':<10} | "
        f"{'Vega':<8} | {'Vega SE':<8} | {'Vega Std':<8} | {'True SE':<8} | {'Naive SE':<8}"
    )
    print("-" * 126)
    for name, _, _ in schemes:
        res = results[name]
        print(
            f"{name:<28} | {res['Price']:<8.4f} | {res['Delta']:<8.4f} | "
            f"{res['Gamma']:<10.6f} | {res['Vega']:<8.4f} | {res['Vega SE']:<8.4f} | "
            f"{res['Vega Std']:<8.4f} | {res['True SE']:<8.4f} | {res['Naive SE']:<8.4f}"
        )
    print("-" * 126)

    # 3. Vega Difference Significance Tests
    print("\n" + "-" * 38 + " Vega Difference Significance Test " + "-" * 38)
    print(
        f"{'Comparison (A vs B)':<46} | {'Diff':<9} | {'Comb SE':<9} | {'t-stat':<8} | {'Paired t':<8} | {'Significance'}"
    )
    print("-" * 115)

    comp_pairs = [
        ("Heston FT (None)", "Heston QE (EMS)"),
        ("Heston FT (None)", "Heston QE (Andersen M-Corr)"),
        ("Heston QE (EMS)", "Heston QE (Andersen M-Corr)"),
    ]

    for name_a, name_b in comp_pairs:
        vegas_a = np.array(collected[name_a]["vegas"])
        vegas_b = np.array(collected[name_b]["vegas"])

        mean_a = float(np.mean(vegas_a))
        mean_b = float(np.mean(vegas_b))
        diff = mean_a - mean_b

        # Independent two-sample SE
        se_a = results[name_a]["Vega SE"]
        se_b = results[name_b]["Vega SE"]
        comb_se = float(np.sqrt(se_a ** 2 + se_b ** 2))
        t_stat = diff / comb_se if comb_se > 0 else np.nan

        # Paired t-test using Common Random Numbers (CRN) across same seeds
        pair_diffs = vegas_a - vegas_b
        pair_se = float(np.std(pair_diffs, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
        paired_t = float(np.mean(pair_diffs) / pair_se) if pair_se > 0 else np.nan

        if abs(t_stat) > 3.29:
            sig = "p < 0.001 (Highly Sig.)"
        elif abs(t_stat) > 1.96:
            sig = "p < 0.05 (Sig.)"
        else:
            sig = "p >= 0.05 (Not Sig. / Noise)"

        comp_label = f"{name_a} - {name_b}"
        print(
            f"{comp_label:<46} | {diff:<9.4f} | {comb_se:<9.4f} | {t_stat:<8.3f} | {paired_t:<8.3f} | {sig}"
        )
    print("-" * 115)

    # 4. Forward Drift Bias Aggregation & Plotting (Plotting currently turned off)
    run_plotting = False
    if run_plotting:
        drift_biases = {}
        drift_biases_se = {}
        for name, _, _ in schemes:
            seed_drift_biases_arr = np.array(collected[name]["seed_drift_biases"])
            drift_biases[name] = np.mean(seed_drift_biases_arr, axis=0)
            drift_biases_se[name] = (
                np.std(seed_drift_biases_arr, axis=0, ddof=1) / np.sqrt(n_seeds)
                if n_seeds > 1
                else np.zeros_like(drift_biases[name])
            )

        palette = {
            "Heston FT (None)": {
                "line": "#0F385A",
                "bound": "#72B3DF",
            },
            "Heston QE (EMS)": {
                "line": "#B23A22",
                "bound": None,
            },
            "Heston QE (Andersen M-Corr)": {
                "line": "#145A32",
                "bound": "#7DC99E",
            },
        }

        plt.figure(figsize=(10, 5), dpi=150)

        for name, _, _ in schemes:
            colors = palette.get(name, {"line": "#333333", "bound": None})
            line_color = colors["line"]
            bound_color = colors["bound"]

            if bound_color is not None and name in drift_biases_se:
                lower_bound = drift_biases[name] - 1.96 * drift_biases_se[name]
                upper_bound = drift_biases[name] + 1.96 * drift_biases_se[name]
                plt.fill_between(
                    time_grid,
                    lower_bound,
                    upper_bound,
                    color=bound_color,
                    edgecolor=line_color,
                    linestyle="--",
                    linewidth=0.75,
                    alpha=0.28,
                    zorder=2,
                )

            plt.plot(
                time_grid,
                drift_biases[name],
                label=name,
                color=line_color,
                lw=2.0,
                zorder=3,
            )

        plt.axhline(
            0.0,
            color="#2C3E50",
            linestyle="--",
            lw=1.2,
            alpha=0.8,
            label="Zero Drift Bias",
            zorder=4,
        )

        plt.title(
            f"Forward Drift Bias: Mean Spot vs Theoretical Forward (T={T}, N={n_sims:,})",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Time $t$ (Years)")
        plt.ylabel(r"$\mathbb{E}[S_t] - S_0 e^{(r-q)t}$")
        plt.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="#DCDCDC",
            loc="upper left",
        )
        plt.grid(True, linestyle=":", alpha=0.4, color="#888888")
        plt.tight_layout()

        plot_filename = (
            f"drift_correction_{n_seeds}_{n_sims // 1000}k.png"
            if (n_sims % 1000 == 0)
            else f"drift_correction_{n_seeds}_{n_sims}.png"
        )
        plt.savefig(plot_filename, dpi=150)

        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "img")
        if os.path.isdir(assets_dir):
            plt.savefig(os.path.join(assets_dir, plot_filename), dpi=150)

        plt.close()
        print(f"\nForward drift bias chart saved as: {plot_filename}")


if __name__ == "__main__":
    main()

# Detected CPU logical processors: 16
# Using worker processes: 14
# Parameters: kappa=1.7, theta=0.04, xi=0.6, rho=-0.7, v_0=0.09
# Feller Ratio: 0.3778 (Satisfied: False)
# Simulation setup: n_sims=50,000, n_seeds=100, total seeds=300
# --------------------------------------------------------------------------------
# Total tasks to run: 300 across 3 schemes.
# Starting multiprocessing execution...
# Progress: [ 15/300] (  5.0%) | Elapsed:  24.9s | ETA: 473.9s
# Progress: [ 30/300] ( 10.0%) | Elapsed:  37.2s | ETA: 334.9s
# Progress: [ 45/300] ( 15.0%) | Elapsed:  49.3s | ETA: 279.1s
# Progress: [ 60/300] ( 20.0%) | Elapsed:  61.4s | ETA: 245.5s
# Progress: [ 75/300] ( 25.0%) | Elapsed:  73.5s | ETA: 220.6s
# Progress: [ 90/300] ( 30.0%) | Elapsed:  85.8s | ETA: 200.2s
# Progress: [105/300] ( 35.0%) | Elapsed: 102.3s | ETA: 189.9s
# Progress: [120/300] ( 40.0%) | Elapsed: 118.5s | ETA: 177.7s
# Progress: [135/300] ( 45.0%) | Elapsed: 134.9s | ETA: 164.9s
# Progress: [150/300] ( 50.0%) | Elapsed: 150.8s | ETA: 150.8s
# Progress: [165/300] ( 55.0%) | Elapsed: 167.1s | ETA: 136.7s
# Progress: [180/300] ( 60.0%) | Elapsed: 182.8s | ETA: 121.8s
# Progress: [195/300] ( 65.0%) | Elapsed: 199.0s | ETA: 107.1s
# Progress: [210/300] ( 70.0%) | Elapsed: 218.4s | ETA:  93.6s
# Progress: [225/300] ( 75.0%) | Elapsed: 242.0s | ETA:  80.7s
# Progress: [240/300] ( 80.0%) | Elapsed: 262.9s | ETA:  65.7s
# Progress: [255/300] ( 85.0%) | Elapsed: 281.8s | ETA:  49.7s
# Progress: [270/300] ( 90.0%) | Elapsed: 300.4s | ETA:  33.4s
# Progress: [285/300] ( 95.0%) | Elapsed: 320.3s | ETA:  16.9s
# Progress: [300/300] (100.0%) | Elapsed: 328.5s | ETA:   0.0s

# All tasks finished in 328.73 seconds (5.48 minutes)!
# ==========================================================================================

# ------------------------------ Put-Call Parity Check (across seeds) ------------------------------
# Scheme                       | Parity Error   | SE         | t-stat  
# ------------------------------------------------------------------------------------------
# Heston FT (None)             | 0.006671       | 0.013678   | 0.488
# Heston QE (EMS)              | -0.000000      | 0.000000   | -4.976
# Heston QE (Andersen M-Corr)  | -0.005711      | 0.012743   | -0.448
# ------------------------------------------------------------------------------------------

# ------------------------------------------------- Asian Call Option -------------------------------------------------
# Scheme                       | Price    | Delta    | Gamma      | Vega     | Vega Std | True SE  | Naive SE
# -------------------------------------------------------------------------------------------------------------------
# Heston FT (None)             | 12.1082  | 0.6778   | 0.013576   | 42.4015  | 0.4504   | 0.0072   | 0.0672
# Heston QE (EMS)              | 12.0225  | 0.6806   | 0.013541   | 43.4208  | 0.3630   | 0.0042   | 0.0663
# Heston QE (Andersen M-Corr)  | 12.0185  | 0.6805   | 0.013596   | 43.4043  | 0.5622   | 0.0070   | 0.0663
# -------------------------------------------------------------------------------------------------------------------

# Detected CPU logical processors: 16
# Using worker processes: 14
# Parameters: kappa=1.7, theta=0.04, xi=0.35, rho=-0.7, v_0=0.09
# Feller Ratio: 1.1102 (Satisfied: True)
# Simulation setup: n_sims=50,000, n_seeds=100, total seeds=300
# --------------------------------------------------------------------------------
# Total tasks to run: 300 across 3 schemes.
# Starting multiprocessing execution...
# Progress: [ 15/300] (  5.0%) | Elapsed:  25.3s | ETA: 480.6s
# Progress: [ 30/300] ( 10.0%) | Elapsed:  37.6s | ETA: 338.3s
# Progress: [ 45/300] ( 15.0%) | Elapsed:  49.8s | ETA: 282.3s
# Progress: [ 60/300] ( 20.0%) | Elapsed:  61.8s | ETA: 247.1s
# Progress: [ 75/300] ( 25.0%) | Elapsed:  74.0s | ETA: 221.9s
# Progress: [ 90/300] ( 30.0%) | Elapsed:  86.2s | ETA: 201.1s
# Progress: [105/300] ( 35.0%) | Elapsed: 101.7s | ETA: 189.0s
# Progress: [120/300] ( 40.0%) | Elapsed: 116.8s | ETA: 175.2s
# Progress: [135/300] ( 45.0%) | Elapsed: 131.9s | ETA: 161.2s
# Progress: [150/300] ( 50.0%) | Elapsed: 147.5s | ETA: 147.5s
# Progress: [165/300] ( 55.0%) | Elapsed: 164.2s | ETA: 134.3s
# Progress: [180/300] ( 60.0%) | Elapsed: 179.6s | ETA: 119.8s
# Progress: [195/300] ( 65.0%) | Elapsed: 197.1s | ETA: 106.2s
# Progress: [210/300] ( 70.0%) | Elapsed: 215.5s | ETA:  92.4s
# Progress: [225/300] ( 75.0%) | Elapsed: 234.2s | ETA:  78.1s
# Progress: [240/300] ( 80.0%) | Elapsed: 254.9s | ETA:  63.7s
# Progress: [255/300] ( 85.0%) | Elapsed: 272.7s | ETA:  48.1s
# Progress: [270/300] ( 90.0%) | Elapsed: 290.6s | ETA:  32.3s
# Progress: [285/300] ( 95.0%) | Elapsed: 307.2s | ETA:  16.2s
# Progress: [300/300] (100.0%) | Elapsed: 316.3s | ETA:   0.0s

# All tasks finished in 316.51 seconds (5.28 minutes)!
# ==========================================================================================

# ------------------------------ Put-Call Parity Check (across seeds) ------------------------------
# Scheme                       | Parity Error   | SE         | t-stat
# ------------------------------------------------------------------------------------------
# Heston FT (None)             | 0.009972       | 0.014746   | 0.676
# Heston QE (EMS)              | -0.000000      | 0.000000   | -4.853
# Heston QE (Andersen M-Corr)  | -0.006944      | 0.013340   | -0.521
# ------------------------------------------------------------------------------------------

# ------------------------------------------------- Asian Call Option -------------------------------------------------
# Scheme                       | Price    | Delta    | Gamma      | Vega     | Vega Std | True SE  | Naive SE
# -------------------------------------------------------------------------------------------------------------------
# Heston FT (None)             | 12.7171  | 0.6427   | 0.012945   | 43.9962  | 0.4390   | 0.0084   | 0.0759
# Heston QE (EMS)              | 12.7161  | 0.6424   | 0.012901   | 43.9832  | 0.1763   | 0.0039   | 0.0760
# Heston QE (Andersen M-Corr)  | 12.7117  | 0.6423   | 0.013045   | 43.9775  | 0.3771   | 0.0079   | 0.0759
# -------------------------------------------------------------------------------------------------------------------