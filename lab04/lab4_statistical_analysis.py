"""
Lab 4: Statistical Analysis — final simplified version (aligned to grading rubric)
- Clean, modular, PEP 8–friendly
- Correct calculations + clear visuals + concise engineering notes
- Generates required PNGs and a structured report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon


# ============ 1) DATA ============ #
def load_data(file_path, fillna=True):
    """
    Read CSV from ../datasets/ (fallback: local).
    If fillna=True:
      - Numeric NaNs -> column mean
      - Non-numeric NaNs -> column mode (fallback "")
    """
    try:
        df = pd.read_csv(f"../datasets/{file_path}")
    except Exception:
        df = pd.read_csv(file_path)

    if fillna:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].mean())
            else:
                m = df[c].mode(dropna=True)
                df[c] = df[c].fillna(m.iloc[0] if not m.empty else "")
    return df


# ============ 2) DESCRIPTIVE STATS & SHAPE ============ #
def calculate_descriptive_stats(data):
    """
    Return one-row DataFrame with key descriptive stats.
    Accepts Series/array or DataFrame (uses first numeric column).
    """
    if isinstance(data, pd.DataFrame):
        num_cols = data.select_dtypes(include=[np.number]).columns
        s = data[num_cols[0]].dropna()
    else:
        s = pd.to_numeric(pd.Series(data), errors="coerce").dropna()

    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
    return pd.DataFrame([{
        "count": s.size,
        "mean": s.mean(),
        "median": q2,
        "mode": mode_val,
        "std": s.std(ddof=1),
        "variance": s.var(ddof=1),
        "min": s.min(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "max": s.max(),
        "range": s.max() - s.min(),
        "skewness": stats.skew(s, bias=False, nan_policy="omit"),
        "kurtosis": stats.kurtosis(s, fisher=True, bias=False, nan_policy="omit"),
    }])


# ============ 3) VISUALIZATIONS ============ #
def plot_distribution(data, title, save_path=None):
    """
    Histogram + KDE; draw mean/median/mode and ±1σ/±2σ/±3σ bands.
    Clear labels/titles for grading rubric.
    """
    s = data if isinstance(data, pd.Series) else pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    st = calculate_descriptive_stats(s).iloc[0]
    mu, sigma, med, mode_val = st["mean"], st["std"], st["median"], st["mode"]

    plt.figure(figsize=(8, 5))
    sns.histplot(s, bins=20, stat="density", kde=True)
    for v, ls, lab in [(mu, "--", "Mean"), (med, "-.", "Median"), (mode_val, ":", "Mode")]:
        plt.axvline(v, linestyle=ls, linewidth=2, label=lab)
    for k, a in zip([1, 2, 3], [0.15, 0.10, 0.05]):
        plt.axvspan(mu - k * sigma, mu + k * sigma, alpha=a, label=f"±{k}σ")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def fit_distribution(data, distribution_type="normal"):
    """Fit normal/uniform/exponential to a numeric vector and return parameters."""
    s = data if isinstance(data, pd.Series) else pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    dt = distribution_type.lower()
    if dt == "normal":
        mu, sd = norm.fit(s)
        return {"type": "normal", "mean": mu, "std": sd}
    if dt == "uniform":
        loc, scale = uniform.fit(s)
        return {"type": "uniform", "loc": loc, "scale": scale}
    if dt == "exponential":
        loc, scale = expon.fit(s, floc=0)
        return {"type": "exponential", "loc": loc, "scale": scale}
    raise ValueError("Supported: normal, uniform, exponential.")


def plot_distribution_fitting(df, column, fitted_dist=None, save_path=None):
    """
    Overlay fitted Normal PDF on histogram of df[column].
    If column is missing, fall back to first numeric column.
    """
    if isinstance(df, pd.DataFrame):
        if column in df.columns:
            s = df[column].dropna()
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                print("[plot_distribution_fitting] Skipped: no numeric column found.")
                return
            print(f"[plot_distribution_fitting] '{column}' not found. Using '{num_cols[0]}' instead.")
            s = df[num_cols[0]].dropna()
    else:
        s = pd.to_numeric(pd.Series(df), errors="coerce").dropna()

    if fitted_dist is None or fitted_dist.get("type") != "normal":
        mu, sd = norm.fit(s)
    else:
        mu, sd = fitted_dist["mean"], fitted_dist["std"]

    x = np.linspace(s.min(), s.max(), 300)
    plt.figure(figsize=(8, 5))
    sns.histplot(s, bins=20, stat="density")
    plt.plot(x, norm.pdf(x, mu, sd), linewidth=2, label=f"Normal(μ={mu:.2f}, σ={sd:.2f})")
    plt.title("Distribution Fitting")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return {"type": "normal", "mean": mu, "std": sd}


def plot_material_comparison(data, column, group_column, save_path=None):
    """Boxplot + points for group comparison (clear labels)."""
    missing = [c for c in [column, group_column] if c not in data.columns]
    if missing:
        print(f"[plot_material_comparison] Skipped: missing columns {missing}.")
        return
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=group_column, y=column)
    sns.stripplot(data=data, x=group_column, y=column, dodge=True, alpha=0.4)
    plt.title("Material Comparison")
    plt.xlabel(group_column)
    plt.ylabel(column)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_probability_distributions(save_path="probability_distributions.png"):
    """Four-panel comparison: Binomial/Poisson (PMF+CDF), Normal/Exponential (PDF+CDF)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Binomial (counts)
    n, p = 100, 0.05
    ks = np.arange(0, 20)
    axes[0, 0].stem(ks, binom.pmf(ks, n, p), basefmt=" ")
    axes[0, 0].plot(ks, binom.cdf(ks, n, p), linewidth=2)
    axes[0, 0].set_title("Binomial (n=100, p=0.05): PMF & CDF")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Probability")

    # Poisson (arrivals)
    lam = 10
    ks = np.arange(0, 30)
    axes[0, 1].stem(ks, poisson.pmf(ks, lam), basefmt=" ")
    axes[0, 1].plot(ks, poisson.cdf(ks, lam), linewidth=2)
    axes[0, 1].set_title("Poisson (λ=10): PMF & CDF")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Probability")

    # Normal (measurements)
    mu, sd = 250, 15
    x = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
    axes[1, 0].plot(x, norm.pdf(x, mu, sd), linewidth=2, label="PDF")
    axes[1, 0].plot(x, norm.cdf(x, mu, sd), linewidth=2, label="CDF")
    axes[1, 0].set_title("Normal (μ=250, σ=15): PDF & CDF")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Density / Prob.")

    # Exponential (times)
    mean = 1000.0
    lam = 1.0 / mean
    x = np.linspace(0, 4000, 400)
    axes[1, 1].plot(x, lam * np.exp(-lam * x), linewidth=2, label="PDF")
    axes[1, 1].plot(x, 1 - np.exp(-lam * x), linewidth=2, label="CDF")
    axes[1, 1].set_title("Exponential (mean=1000): PDF & CDF")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Density / Prob.")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_statistical_summary_dashboard(df_conc=None, df_mats=None, df_loads=None,
                                       save_path="statistical_summary_dashboard.png"):
    """2×2 dashboard: concrete dist, concrete by mix, first 100 loads, material means."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (1) Concrete distribution
    if df_conc is not None:
        s = df_conc["strength_mpa"].dropna() if "strength_mpa" in df_conc else \
            df_conc.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
        sns.histplot(s, bins=20, stat="density", kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Concrete Strength Distribution")
        axes[0, 0].set_xlabel("MPa")
        axes[0, 0].set_ylabel("Density")
    else:
        axes[0, 0].text(0.5, 0.5, "No concrete data", ha="center")

    # (2) Concrete by mix_type
    if df_conc is not None and {"mix_type", "strength_mpa"}.issubset(df_conc.columns):
        sns.boxplot(data=df_conc, x="mix_type", y="strength_mpa", ax=axes[0, 1])
        axes[0, 1].set_title("Concrete by Mix Type")
        axes[0, 1].set_xlabel("mix_type")
        axes[0, 1].set_ylabel("strength_mpa")
    else:
        axes[0, 1].text(0.5, 0.5, "No mix-type detail", ha="center")

    # (3) Structural loads (first 100)
    if df_loads is not None and "load_kN" in df_loads.columns:
        y = df_loads["load_kN"].head(100).dropna()
        axes[1, 0].plot(np.arange(len(y)), y, linewidth=1.8)
        axes[1, 0].set_title("Structural Loads (first 100 samples)")
        axes[1, 0].set_xlabel("sample")
        axes[1, 0].set_ylabel("load_kN")
    else:
        axes[1, 0].text(0.5, 0.5, "No loads data", ha="center")

    # (4) Material means
    if df_mats is not None and {"material_type", "yield_strength_mpa"}.issubset(df_mats.columns):
        means = df_mats.groupby("material_type")["yield_strength_mpa"].mean().sort_values()
        axes[1, 1].bar(means.index, means.values)
        axes[1, 1].set_title("Material Mean Yield Strength")
        axes[1, 1].set_xlabel("material_type")
        axes[1, 1].set_ylabel("mean (MPa)")
    else:
        axes[1, 1].text(0.5, 0.5, "No materials data", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============ 4) PROBABILITY TOOLS ============ #
def calculate_probability_binomial(n, p, k):
    """Return P(X=k), P(X<=k), P(X>=k) for Binomial(n,p)."""
    pmf = binom.pmf(k, n, p)
    cdf_le = binom.cdf(k, n, p)
    cdf_ge = 1 - binom.cdf(k - 1, n, p) if k > 0 else 1.0
    return {"pmf": pmf, "cdf_le": cdf_le, "cdf_ge": cdf_ge}


def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """Return tail/interval probability for Normal(mean,std)."""
    d = norm(loc=mean, scale=std)
    if x_lower is None and x_upper is None:
        raise ValueError("Provide x_lower or x_upper.")
    if x_lower is None:
        return float(d.cdf(x_upper))
    if x_upper is None:
        return float(1 - d.cdf(x_lower))
    if x_upper < x_lower:
        raise ValueError("x_upper must be ≥ x_lower.")
    return float(d.cdf(x_upper) - d.cdf(x_lower))


def calculate_probability_poisson(lambda_param, k):
    """Return P(X=k), P(X<=k), P(X>k) for Poisson(λ)."""
    pmf = poisson.pmf(k, lambda_param)
    cdf_le = poisson.cdf(k, lambda_param)
    return {"pmf": pmf, "cdf_le": cdf_le, "cdf_gt": 1 - cdf_le}


def calculate_probability_exponential(mean, x):
    """Return PDF/CDF/SF at x for Exponential(mean)."""
    lam = 1.0 / mean
    x = max(0.0, x)
    cdf = 1 - np.exp(-lam * x)
    return {"lambda": lam, "cdf": cdf, "sf": 1 - cdf, "pdf": lam * np.exp(-lam * x)}


def apply_bayes_theorem(prior, sensitivity, specificity):
    """
    Bayes for test interpretation:
      prior = P(D), sensitivity = P(+|D), specificity = P(-|~D)
    Returns PPV = P(D|+), NPV = P(~D|-), and marginals.
    """
    p_d = prior
    p_nd = 1 - p_d
    p_pos = sensitivity * p_d + (1 - specificity) * p_nd
    p_neg = (1 - sensitivity) * p_d + specificity * p_nd
    ppv = (sensitivity * p_d) / p_pos if p_pos > 0 else np.nan
    npv = (specificity * p_nd) / p_neg if p_neg > 0 else np.nan
    return {"P(D)": p_d, "P(~D)": p_nd, "P(Pos)": p_pos, "P(Neg)": p_neg, "PPV": ppv, "NPV": npv}


def plot_probability_tree(prior, sensitivity, specificity, save_path="bayes_probability_tree.png"):
    """Creates a visual probability tree for the Bayes' theorem scenario."""
    
    p_no_damage = 1 - prior
    p_false_pos = 1 - specificity
    p_true_neg = specificity
    p_false_neg = 1 - sensitivity
    
    p_test_given_damage = sensitivity
    p_damage = prior
    p_test_given_no_damage = p_false_pos 
    
    p_test_pos = (p_test_given_damage * p_damage) + (p_test_given_no_damage * p_no_damage)
    
    if p_test_pos == 0: 
        posterior_prob = np.nan
    else:
        posterior_prob = (p_test_given_damage * p_damage) / p_test_pos

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off') 
    
    # Level 0
    ax.text(0.0, 0.5, "Population", ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="circle,pad=0.3", fc="lightblue", ec="b"))
    # Level 1
    ax.text(0.2, 0.8, f"P(Damage) = {prior:.2f}", ha='left', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="r"))
    ax.text(0.2, 0.2, f"P(No Damage) = {p_no_damage:.2f}", ha='left', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", ec="g"))
    # Arrows 0->1
    ax.annotate("", xy=(0.18, 0.78), xytext=(0.05, 0.53), arrowprops=dict(arrowstyle="->", ec="r", lw=2))
    ax.annotate("", xy=(0.18, 0.22), xytext=(0.05, 0.47), arrowprops=dict(arrowstyle="->", ec="g", lw=2))
    # Level 2
    ax.text(0.5, 0.9, f"P(+Test | Damage) = {sensitivity:.2f} (True Pos)", ha='left', va='center', fontsize=11)
    ax.text(0.5, 0.7, f"P(-Test | Damage) = {p_false_neg:.2f} (False Neg)", ha='left', va='center', fontsize=11)
    ax.text(0.5, 0.3, f"P(+Test | No Damage) = {p_false_pos:.2f} (False Pos)", ha='left', va='center', fontsize=11)
    ax.text(0.5, 0.1, f"P(-Test | No Damage) = {p_true_neg:.2f} (True Neg)", ha='left', va='center', fontsize=11)
    # Arrows 1->2
    ax.annotate("", xy=(0.48, 0.9), xytext=(0.35, 0.8), arrowprops=dict(arrowstyle="->", ec="gray", lw=1))
    ax.annotate("", xy=(0.48, 0.7), xytext=(0.35, 0.8), arrowprops=dict(arrowstyle="->", ec="gray", lw=1))
    ax.annotate("", xy=(0.48, 0.3), xytext=(0.35, 0.2), arrowprops=dict(arrowstyle="->", ec="gray", lw=1))
    ax.annotate("", xy=(0.48, 0.1), xytext=(0.35, 0.2), arrowprops=dict(arrowstyle="->", ec="gray", lw=1))
    
    # Result
    ax.text(0.5, 0.5, 
            f"Posterior Probability (Given +Test):\n\n"
            f"P(Damage | +Test) = {posterior_prob:.4f} ({posterior_prob*100:.2f}%)",
            ha='left', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=1", fc="gold", ec="orange", lw=2))

    plt.title("Bayes' Theorem: Probability Tree Diagram", fontsize=16, y=1.05)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Probability tree diagram saved to {save_path}")
    plt.show()
    plt.close()


def plot_material_comparison(data, column, group_column, save_path=None):
    missing = [c for c in [column, group_column] if c not in data.columns]
    if missing:
        print(f"[plot_material_comparison] Skipped: missing columns {missing}. Available: {list(data.columns)}")
        return
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=group_column, y=column)
    sns.stripplot(data=data, x=group_column, y=column, dodge=True, alpha=0.4)
    plt.title("Material Comparison"); plt.xlabel(group_column); plt.ylabel(column)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()
    plt.close()

# ============ 5) REPORT ============ #
def create_statistical_report(data, output_file="lab4_statistical_report.txt"):
    """
    Structured report for grading rubric:
      1) Descriptive statistics table
      2) Distribution parameters (Normal fit)
      3) Key findings & interpretations
      4) Engineering implications
    """
    s = data if isinstance(data, pd.Series) else pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    if s.empty:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Lab 4 - Statistical Report\n" + "-" * 40 + "\nNo valid numeric data.\n")
        return output_file

    st = calculate_descriptive_stats(s).round(4)
    row = st.iloc[0]
    mu, sigma = norm.fit(s)
    cov = (sigma / mu) if mu != 0 else np.nan
    p05 = norm.ppf(0.05, mu, sigma)
    p95 = norm.ppf(0.95, mu, sigma)

    skew_ = float(row["skewness"])
    kurt_ = float(row["kurtosis"])
    q1, q3, iqr_ = float(row["Q1"]), float(row["Q3"]), float(row["IQR"])
    low_fence, high_fence = q1 - 1.5 * iqr_, q3 + 1.5 * iqr_
    outliers = int(((s < low_fence) | (s > high_fence)).sum())

    skew_txt = ("approximately symmetric" if abs(skew_) < 0.3
                else "mild right-skew" if skew_ > 0 else "mild left-skew")
    if abs(skew_) >= 0.8:
        skew_txt = "strong right-skew" if skew_ > 0 else "strong left-skew"
    kurt_txt = ("near-normal tails" if abs(kurt_) < 0.3
                else "heavier-than-normal tails" if kurt_ > 0.3 else "lighter-than-normal tails")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Lab 4 - Statistical Report\n")
        f.write("-" * 40 + "\n\n")

        f.write("1) Descriptive Statistics\n")
        f.write(st.to_string(index=False))
        f.write("\n\n")

        f.write("2) Distribution Parameters (Normal fit)\n")
        f.write(f"   mean (μ): {mu:.4f}\n")
        f.write(f"   std  (σ): {sigma:.4f}\n")
        f.write(f"   CoV  (σ/μ): {cov:.4f}\n")
        f.write(f"   5th percentile: {p05:.4f}\n")
        f.write(f"   95th percentile: {p95:.4f}\n\n")

        f.write("3) Key Findings & Interpretations\n")
        f.write(f"   • Shape: skewness={skew_:.4f} ({skew_txt}), kurtosis={kurt_:.4f} ({kurt_txt})\n")
        f.write(f"   • Spread: std={row['std']:.4f}, IQR={iqr_:.4f}, range={row['range']:.4f}\n")
        f.write(f"   • Data span: min={row['min']:.4f}, max={row['max']:.4f}, outliers (IQR rule)={outliers}\n")
        f.write("   • Visual check: histogram/KDE and overlayed Normal curve were generated.\n\n")

        f.write("4) Engineering Implications\n")
        f.write("   • Use the 5th percentile as a conservative characteristic value in design checks.\n")
        if np.isnan(cov):
            f.write("   • CoV undefined (μ≈0). Consider data consistency before design decisions.\n")
        elif cov < 0.05:
            f.write("   • Low variability (CoV<0.05): process is highly consistent; standard QC suffices.\n")
        elif cov < 0.15:
            f.write("   • Moderate variability: maintain routine QC; consider monitoring mix/material changes.\n")
        else:
            f.write("   • High variability: tighten process controls and consider larger safety factors.\n")
        f.write("   • Investigate outliers for measurement errors or batch effects; group-wise analysis may help.\n")
    return output_file


# ============ 6) MAIN (ties everything together) ============ #
def main():
    # --- Load datasets ---
    conc = mats = loads = None
    try:
        conc = load_data("concrete_strength.csv")
    except Exception as e:
        print("Concrete load warning:", e)
    try:
        mats = load_data("material_properties.csv")
    except Exception as e:
        print("Materials load warning:", e)
    try:
        loads = load_data("structural_loads.csv")
    except Exception as e:
        print("Loads load warning:", e)

    # --- Descriptive Statistics (Console) ---
    if conc is not None:
        s_conc = conc["strength_mpa"] if "strength_mpa" in conc else \
                 conc.select_dtypes(include=[np.number]).iloc[:, 0]
        print("\n=== Summary Statistics (Concrete Strength) ===")
        print(calculate_descriptive_stats(s_conc).round(3).to_string(index=False))

    if loads is not None and "load_kN" in loads.columns:
        print("\n=== Summary Statistics (Structural Loads) ===")
        print(calculate_descriptive_stats(loads["load_kN"]).round(3).to_string(index=False))

    # --- Distribution Fitting (Console + PNG) ---
    if conc is not None:
        fit_params = fit_distribution(s_conc, "normal")
        print("\n=== Distribution Parameters (Concrete ~ Normal) ===")
        print(pd.DataFrame([fit_params]).to_string(index=False))

    # --- Comparative Statistics (Console) ---
    if mats is not None and {"material_type", "yield_strength_mpa"}.issubset(mats.columns):
        comp = mats.groupby("material_type")["yield_strength_mpa"].agg(
            count="count", mean="mean", std="std", min="min", max="max"
        ).round(3)
        print("\n=== Comparative Statistics (Material Strength by material_type) ===")
        print(comp.to_string())

    if conc is not None and {"mix_type", "strength_mpa"}.issubset(conc.columns):
        comp_conc = conc.groupby("mix_type")["strength_mpa"].agg(
            count="count", mean="mean", std="std", min="min", max="max"
        ).round(3)
        print("\n=== Comparative Statistics (Concrete Strength by mix_type) ===")
        print(comp_conc.to_string())

    # --- Probability Calculations (Console) ---
    print("\n=== Probability Calculations ===")
    print(f"Binomial n=100, p=0.05 -> P(X=3)={calculate_probability_binomial(100,0.05,3)['pmf']:.6f}, "
          f"P(X≤5)={binom.cdf(5,100,0.05):.6f}")
    print(f"Poisson λ=10 -> P(X=8)={calculate_probability_poisson(10,8)['pmf']:.6f}, "
          f"P(X>15)={1 - poisson.cdf(15,10):.6f}")
    print(f"Normal μ=250, σ=15 -> P(X>280)={calculate_probability_normal(250,15,x_lower=280):.6f}, "
          f"95th percentile={norm.ppf(0.95,250,15):.3f}")
    print(f"Exponential mean=1000 -> P(fail<500)={calculate_probability_exponential(1000,500)['cdf']:.6f}, "
          f"P(survive>1500)={calculate_probability_exponential(1000,1500)['sf']:.6f}")

    # --- Bayes' Theorem (Console + PNG) ---
    bayes = apply_bayes_theorem(prior=0.05, sensitivity=0.95, specificity=0.90)
    print("\n=== Bayes' Theorem (Damage Detection) ===")
    print(f"P(D)={bayes['P(D)']:.3f}, P(~D)={bayes['P(~D)']:.3f}")
    print(f"P(Pos)={bayes['P(Pos)']:.3f}, P(Neg)={bayes['P(Neg)']:.3f}")
    print(f"PPV=P(D|Pos)={bayes['PPV']:.3f}, NPV=P(~D|Neg)={bayes['NPV']:.3f}")
    plot_probability_tree(0.05, 0.95, 0.90, save_path="bayes_probability_tree.png")

    # --- PNGs required by spec ---
    if conc is not None:
        plot_distribution(s_conc, "Concrete Strength", "concrete_strength_distribution.png")
        plot_distribution_fitting(conc, column="strength_mpa", save_path="distribution_fitting.png")
        create_statistical_report(s_conc)  # lab4_statistical_report.txt

    if mats is not None and {"material_type", "yield_strength_mpa"}.issubset(mats.columns):
        plot_material_comparison(mats, "yield_strength_mpa", "material_type",
                                 "material_comparison_boxplot.png")

    plot_probability_distributions("probability_distributions.png")
    plot_statistical_summary_dashboard(df_conc=conc, df_mats=mats, df_loads=loads,
                                       save_path="statistical_summary_dashboard.png")


if __name__ == "__main__":
    main()





