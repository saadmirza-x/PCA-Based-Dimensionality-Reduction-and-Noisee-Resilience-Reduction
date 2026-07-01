# =============================================================================
# CMT Semester Project — CE342 Spring 2026
# =============================================================================
# Dataset   : Country Data (HELP International)
# Team      : Saad Mirza (2023498) | Moiz ud din (2023315) | Hassan Khalid (2023435)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # required for 3D scatter plots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# -----------------------------------------------------------------------------
# REPRODUCIBILITY — fixed seed so every run gives identical results
# -----------------------------------------------------------------------------
np.random.seed(42)


# =============================================================================
# TASK 1  (CLO 1 | 10 Marks)
# Arrange dataset into proper matrix-vector form.
# Add Gaussian noise using the dynamically computed SNR value.
# =============================================================================

# --- Load dataset ------------------------------------------------------------
df  = pd.read_csv('Country-data.csv')
dc  = df.drop('country', axis=1)      # numeric features only
cld = dc.to_numpy()                   # raw matrix: (n_countries x n_features)

print("=" * 65)
print("TASK 1 — DATA SETUP")
print("=" * 65)
print(f"Features ({dc.shape[1]}): {dc.columns.tolist()}")
print(f"Dataset shape : {cld.shape[0]} countries  x  {cld.shape[1]} features")
print()
print(df.head())

# --- Dynamic SNR Calculation -------------------------------------------------
reg_last3 = [498, 315, 435]
avg_id    = int(sum(reg_last3) / len(reg_last3))          # 416
snr_value = sum(int(digit) for digit in str(avg_id))      # 4+1+6 = 11

print(f"\nRegistration last-3 digits : {reg_last3}")
print(f"Average                    : sum={sum(reg_last3)} / {len(reg_last3)} = {avg_id}")
print(f"Digit sum (SNR)            : {' + '.join(list(str(avg_id)))} = {snr_value}")

# --- Standardise FIRST (statistically correct noise addition) ----------------
mean_raw   = np.mean(cld, axis=0)
std_raw    = np.std(cld,  axis=0)
cld_prestd = (cld - mean_raw) / std_raw          # shape: (167, 9), mean≈0, std≈1

# --- Gaussian Noise Generation -----------------------------------------------
signal_power = np.var(cld_prestd, axis=0)        # ≈ 1.0 per feature
noise_power  = signal_power / snr_value          # noise power per feature
noise_matrix = np.random.normal(0, np.sqrt(noise_power), cld_prestd.shape)
x_noise_prestd = cld_prestd + noise_matrix        # standardised noisy data

print(f"\nSNR value                  : {snr_value}")
print(f"\nClean  matrix shape (standardised) : {cld_prestd.shape}")
print(f"Noisy  matrix shape (standardised) : {x_noise_prestd.shape}")
print(f"Signal power per feature (≈1.0)    : {signal_power.round(4)}")
print(f"Noise  power per feature           : {noise_power.round(4)}")


# =============================================================================
# TASK 2a  (CLO 2 | 20 Marks)
# Implement PCA from scratch.
# Test on a small dataset and verify against sklearn.
# =============================================================================

print("\n" + "=" * 65)
print("TASK 2a — CUSTOM PCA IMPLEMENTATION & VERIFICATION")
print("=" * 65)

# --- Custom PCA Function -----------------------------------------------------
def custom_pca(X, num_components, standardize=True):
    """
    PCA from scratch — 6 steps:
    Includes a boolean toggle to prevent redundant scaling of already-standardised data.
    """
    if standardize:
        mean_vals = np.mean(X, axis=0)
        std_vals  = np.std(X,  axis=0)
        X_std     = (X - mean_vals) / std_vals
    else:
        # If already standardised upstream, mean is effectively 0 and std is 1
        mean_vals = np.zeros(X.shape[1])
        std_vals  = np.ones(X.shape[1])
        X_std     = X

    cov_matrix              = np.cov(X_std.T)
    eigenvalues, eigenvecs  = np.linalg.eigh(cov_matrix)

    idx              = np.argsort(eigenvalues)[::-1]
    sorted_eigenvals = eigenvalues[idx]
    sorted_eigvecs   = eigenvecs[:, idx]

    top_eigvecs = sorted_eigvecs[:, :num_components]
    X_reduced   = X_std @ top_eigvecs

    return X_reduced, sorted_eigenvals, top_eigvecs, mean_vals, std_vals


# --- Reconstruction & MSE helpers --------------------------------------------
def reconstruct_data(X_reduced, top_eigvecs, mean_vals, std_vals):
    X_rec_std = X_reduced @ top_eigvecs.T
    return (X_rec_std * std_vals) + mean_vals

def compute_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


# --- Test on small 3x2 dataset -----------------------------------------------
test_data = np.array([[1, 2],
                      [5, 4],
                      [8, 3]], dtype=float)

# Test data is raw, so standardize=True is required
test_reduced, test_evals, test_evecs, test_mean, test_std = custom_pca(test_data, num_components=1, standardize=True)

scaler_test  = StandardScaler()
test_scaled  = scaler_test.fit_transform(test_data)
sk_pca_small = PCA(n_components=1)
sk_result    = sk_pca_small.fit_transform(test_scaled)

print(f"\nSmall dataset shape : {test_data.shape}  →  reduced: {test_reduced.shape}")
print(f"Custom PCA result   :\n{test_reduced}")
print(f"Sklearn PCA result  :\n{sk_result}")

magnitude_match_small = np.allclose(np.abs(test_reduced), np.abs(sk_result))
print(f"Magnitude match (small dataset) : {magnitude_match_small}")


# =============================================================================
# TASK 2b  (CLO 2 | 40 Marks)
# Apply PCA to country dataset (clean standardised + noisy standardised).
# =============================================================================

print("\n" + "=" * 65)
print("TASK 2b — PCA ON COUNTRY DATASET (CLEAN & NOISY)")
print("=" * 65)

k_components = 2    # k=2 << n=9  satisfies the k << n requirement

# --- Apply custom PCA (k=2 for analysis) -------------------------------------
# Data was already standardised in Task 1, so toggle standardise=False
cld_reduced,   cld_evals,   cld_evecs,   cld_mn,   cld_sd   = custom_pca(cld_prestd,   k_components, standardize=False)
noise_reduced, noise_evals, noise_evecs, noise_mn, noise_sd = custom_pca(x_noise_prestd, k_components, standardize=False)

print(f"\nOriginal feature count  (n) : {cld_prestd.shape[1]}")
print(f"Reduced component count (k) : {k_components}   [k << n  ✓]")
print(f"Clean  reduced shape        : {cld_reduced.shape}")
print(f"Noisy  reduced shape        : {noise_reduced.shape}")

# --- Apply custom PCA (k=3 for 3D visualisation only) ------------------------
cld_reduced_3d,   _, _, _, _ = custom_pca(cld_prestd,    3, standardize=False)
noise_reduced_3d, _, _, _, _ = custom_pca(x_noise_prestd, 3, standardize=False)

# --- sklearn verification on FULL country dataset ----------------------------
sk_pca_full  = PCA(n_components=k_components)
sk_full_res  = sk_pca_full.fit_transform(cld_prestd)
full_match   = np.allclose(np.abs(cld_reduced), np.abs(sk_full_res))

print(f"\nMagnitude match vs sklearn (full country dataset) : {full_match}")

# --- Reconstruction ----------------------------------------------------------
cld_reconstructed   = reconstruct_data(cld_reduced,   cld_evecs,   cld_mn, cld_sd)
noise_reconstructed = reconstruct_data(noise_reduced, noise_evecs, noise_mn, noise_sd)

mse_clean   = compute_mse(cld_prestd,     cld_reconstructed)
mse_noisy   = compute_mse(x_noise_prestd, noise_reconstructed)
mse_denoise = compute_mse(cld_prestd,     noise_reconstructed)
mse_raw_noise = compute_mse(cld_prestd,   x_noise_prestd)

print(f"\n--- Reconstruction MSE (in standardised space) ---")
print(f"MSE  clean reconstructed  vs  clean original  : {mse_clean:.6f}  ← dim reduction loss")
print(f"MSE  noisy reconstructed  vs  noisy original  : {mse_noisy:.6f}  ← noisy reconstruction")
print(f"MSE  noisy reconstructed  vs  clean original  : {mse_denoise:.6f}  ← denoising metric")
print(f"MSE  raw noisy data       vs  clean original  : {mse_raw_noise:.6f}  ← noise baseline")
print(f"MSE increase due to noise                     : {((mse_noisy - mse_clean)/mse_clean)*100:.2f}%")
print(f"PCA denoising improvement                     : {((mse_raw_noise - mse_denoise)/mse_raw_noise)*100:.2f}%")

# --- Explained Variance Ratio ------------------------------------------------
cld_var_ratio    = cld_evals   / np.sum(cld_evals)
noise_var_ratio  = noise_evals / np.sum(noise_evals)
cld_cumulative   = np.cumsum(cld_var_ratio)
noise_cumulative = np.cumsum(noise_var_ratio)

var_retained_clean = cld_var_ratio[:k_components].sum()   * 100
var_retained_noisy = noise_var_ratio[:k_components].sum() * 100
print(f"\n--- Variance Retained at k={k_components} ---")
print(f"Clean data : {var_retained_clean:.2f}%")
print(f"Noisy data : {var_retained_noisy:.2f}%")

# --- Optimal k (90% variance threshold) -------------------------------------
threshold        = 0.90
optimal_k_clean  = int(np.argmax(cld_cumulative   >= threshold) + 1)
optimal_k_noisy  = int(np.argmax(noise_cumulative >= threshold) + 1)
print(f"\n--- Optimal k for >= {int(threshold*100)}% Variance ---")
print(f"Clean data optimal k : {optimal_k_clean}")
print(f"Noisy data optimal k : {optimal_k_noisy}")

# --- Eigenvalue Comparison ---------------------------------------------------
n_show = min(5, len(cld_evals))
print(f"\n--- Top-{n_show} Eigenvalue Comparison (Clean vs Noisy) ---")
print(f"{'Component':<12} {'Clean EV':>12} {'Noisy EV':>12} {'Difference':>12}")
for i in range(n_show):
    ce, ne = cld_evals[i], noise_evals[i]
    print(f"  PC{i+1:<9} {ce:>12.4f} {ne:>12.4f} {ne-ce:>+12.4f}")

# ADD THIS after Task 2b PCA
print("\n--- PC Loadings (Top 2 Eigenvectors) ---")
feature_names = dc.columns.tolist()
for i in range(k_components):
    print(f"\nPC{i+1} loadings:")
    for fname, loading in zip(feature_names, cld_evecs[:, i]):
        print(f"  {fname:<20}: {loading:+.4f}")

# =============================================================================
# TASK 3a  (CLO 3 | 10 Marks)
# Present results using the most appropriate tools.
# Five plots: Explained Variance, Scree, 2D Scatter, 3D Scatter, MSE Bar.
# =============================================================================

print("\n" + "=" * 65)
print("TASK 3a — VISUALISATION (5 plots)")
print("=" * 65)

countries = df['country'].values
n_comp    = len(cld_evals)
comp_x    = range(1, n_comp + 1)

# --------------------------------------------------------------------------
# PLOT 1 — Explained Variance Ratio (Individual + Cumulative)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Explained Variance Ratio — Clean vs Noisy', fontsize=13, fontweight='bold')

for ax, var_ratio, cumulative, title, c1, c2 in [
    (axes[0], cld_var_ratio,   cld_cumulative,   'Clean Data (SNR=∞)',          'steelblue', 'navy'),
    (axes[1], noise_var_ratio, noise_cumulative, f'Noisy Data (SNR={snr_value})', 'tomato',    'darkred'),
]:
    ax.plot(comp_x, var_ratio,  marker='o', linestyle='--', color=c1, label='Individual')
    ax.plot(comp_x, cumulative, marker='s', linestyle='-',  color=c2, label='Cumulative')
    ax.axhline(y=threshold, color='gray', linestyle=':', linewidth=1.2,
               label=f'{int(threshold*100)}% threshold')
    ax.set_title(title); ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Ratio'); ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig('plot1_explained_variance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot1_explained_variance.png")

# --------------------------------------------------------------------------
# PLOT 2 — Scree Plot (eigenvalue bar chart with optimal-k marker)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Scree Plot — Eigenvalues per Component', fontsize=13, fontweight='bold')

for ax, evals, opt_k, title, color in [
    (axes[0], cld_evals,   optimal_k_clean, 'Clean Data',                  'steelblue'),
    (axes[1], noise_evals, optimal_k_noisy, f'Noisy Data (SNR={snr_value})', 'tomato'),
]:
    ax.bar(comp_x, evals, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(x=opt_k, color='black', linestyle='--', linewidth=1.5,
               label=f'Optimal k={opt_k}  (≥{int(threshold*100)}% var)')
    ax.set_title(title); ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue'); ax.legend(); ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig('plot2_scree.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot2_scree.png")

# --------------------------------------------------------------------------
# PLOT 3 — 2D PCA Scatter (Clean vs Noisy)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('2D PCA Projection — Cluster Comparison', fontsize=13, fontweight='bold')


for ax, reduced, title, color, var_ratio in [
    (axes[0], cld_reduced,   'Clean Data (k=2)',              'steelblue', cld_var_ratio),
    (axes[1], noise_reduced, f'Noisy Data (k=2, SNR={snr_value})', 'tomato', noise_var_ratio),
]:
    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, color=color, s=40)
    for i in range(0, len(countries), 10):
        ax.text(reduced[i, 0], reduced[i, 1], countries[i], fontsize=8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% variance)')
    ax.grid(True)

plt.tight_layout()
plt.savefig('plot3_scatter_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot3_scatter_2d.png")

# --------------------------------------------------------------------------
# PLOT 4 — 3D PCA Scatter (Clean vs Noisy) — k=3 projection
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6))
fig.suptitle('3D PCA Projection — Cluster Comparison', fontsize=13, fontweight='bold')

for idx_plot, (reduced_3d, title, color) in enumerate([
    (cld_reduced_3d,   'Clean Data (k=3)',              'steelblue'),
    (noise_reduced_3d, f'Noisy Data (k=3, SNR={snr_value})', 'tomato'),
], start=1):
    ax = fig.add_subplot(1, 2, idx_plot, projection='3d')
    ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2],
               alpha=0.7, color=color, s=30)
    for i in range(0, len(countries), 15):
        ax.text(reduced_3d[i, 0], reduced_3d[i, 1], reduced_3d[i, 2],
                countries[i], fontsize=7, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2'); ax.set_zlabel('PC 3')

plt.tight_layout()
plt.savefig('plot4_scatter_3d.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot4_scatter_3d.png")

# --------------------------------------------------------------------------
# PLOT 5 — MSE Comparison Bar Chart (all four MSE metrics)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
labels = [
    'Clean\nrecon vs clean\n(dim reduction loss)',
    'Noisy\nrecon vs noisy\n(noisy reconstruction)',
    'Noisy recon\nvs clean\n(denoising metric)',
    'Raw noisy\nvs clean\n(noise baseline)',
]
values = [mse_clean, mse_noisy, mse_denoise, mse_raw_noise]
colors = ['steelblue', 'tomato', 'darkorange', 'gray']

bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.55)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f'{val:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('Reconstruction MSE — Full Comparison', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Squared Error (standardised space)')
ax.grid(True, axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('plot5_mse_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot5_mse_comparison.png")


# =============================================================================
# TASK 3b  (CLO 3 | 20 Marks)
# Quantitative comparison summary — numerical basis for report discussion.
# =============================================================================



fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(np.corrcoef(cld_prestd.T), annot=True, fmt='.2f',
            xticklabels=dc.columns, yticklabels=dc.columns,
            cmap='coolwarm', center=0, ax=ax)
ax.set_title('Feature Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('plot0_correlation_heatmap.png', dpi=150, bbox_inches='tight')

print("\n" + "=" * 65)
print("TASK 3b — COMPARISON & PERFORMANCE ANALYSIS SUMMARY")
print("=" * 65)

ev_spread_clean = cld_evals[0]   / cld_evals[-1]
ev_spread_noisy = noise_evals[0] / noise_evals[-1]

print(f"""
┌──────────────────────────────────────────────────────────────────┐
│                PCA PERFORMANCE COMPARISON SUMMARY                │
├───────────────────────────────┬───────────────┬─────────────────┤
│  Metric                       │  Clean Data   │   Noisy Data    │
├───────────────────────────────┼───────────────┼─────────────────┤
│  Variance retained  (k=2)     │  {var_retained_clean:>9.2f}%   │  {var_retained_noisy:>9.2f}%       │
│  Reconstruction MSE           │  {mse_clean:>11.6f}   │  {mse_noisy:>11.6f}    │
│  Denoising MSE (recon vs cln) │  {'N/A':>11}   │  {mse_denoise:>11.6f}    │
│  Optimal k  (≥90% threshold)  │  {optimal_k_clean:>11}   │  {optimal_k_noisy:>11}    │
│  PC1 eigenvalue               │  {cld_evals[0]:>11.4f}   │  {noise_evals[0]:>11.4f}    │
│  Eigenvalue spread (PC1/PC9)  │  {ev_spread_clean:>11.2f}   │  {ev_spread_noisy:>11.2f}    │
├───────────────────────────────┴───────────────┴─────────────────┤
│  Raw noisy vs clean MSE (pre-PCA baseline) :  {mse_raw_noise:.6f}          │
│  PCA denoising improvement                 :  {((mse_raw_noise-mse_denoise)/mse_raw_noise)*100:.2f}%               │
│  MSE increase due to noise                 :  {((mse_noisy-mse_clean)/mse_clean)*100:.2f}%               │
└──────────────────────────────────────────────────────────────────┘

KEY INFERENCES FOR REPORT (Task 3b):
──────────────────────────────────────────────────────────────────
1. NOISE SPREADS EIGENVALUE DISTRIBUTION
   Clean PC1 eigenvalue ({cld_evals[0]:.4f}) vs Noisy ({noise_evals[0]:.4f}).
   Noise inflates variance uniformly across all components, compressing
   the eigenvalue spread from {ev_spread_clean:.2f}x (clean) to {ev_spread_noisy:.2f}x (noisy).
   This means PCA can no longer identify a single dominant direction
   as cleanly — energy is diffused across all 9 components.

2. VARIANCE RETENTION DEGRADES UNDER NOISE
   k=2 retains {var_retained_clean:.2f}% variance on clean data vs {var_retained_noisy:.2f}% on noisy data.
   Noise corrupts the compact correlated structure PCA exploits.
   More components are needed post-noise to capture the same
   amount of signal: optimal k shifts from {optimal_k_clean} → {optimal_k_noisy}.

3. RECONSTRUCTION MSE INCREASES WITH NOISE
   MSE rises from {mse_clean:.6f} (clean) to {mse_noisy:.6f} (noisy) —
   a {((mse_noisy-mse_clean)/mse_clean)*100:.2f}% increase. The added noise introduces information
   that PCA cannot represent in k=2 dimensions, forcing higher
   reconstruction error when projecting back to 9D.

4. PCA ACTS AS A PARTIAL DENOISER
   Denoising MSE ({mse_denoise:.6f}) < raw noise baseline ({mse_raw_noise:.6f}).
   PCA's dimensionality reduction filters out {((mse_raw_noise-mse_denoise)/mse_raw_noise)*100:.2f}% of the noise
   energy by discarding low-eigenvalue components that primarily
   capture noise rather than signal structure.

5. CLUSTER SEPARATION DEGRADES IN 2D AND 3D
   The 2D and 3D scatter plots show that clean data forms tighter,
   more distinct country clusters (e.g. developed vs developing
   nations visible along PC1). Noisy scatter plots show cluster
   blurring, with countries shifting position and outliers migrating
   closer to the cluster centroids — noise homogenises the dataset.
──────────────────────────────────────────────────────────────────
""")

print("ALL PLOTS SAVED:")
print("  plot1_explained_variance.png")
print("  plot2_scree.png")
print("  plot3_scatter_2d.png")
print("  plot4_scatter_3d.png")
print("  plot5_mse_comparison.png")
print()
print("[DONE] ALL Task 1 / 2a / 2b / 3a / 3b requirements fulfilled.")
print("       Dynamic SNR | Pre-standardised noise | 2D + 3D plots | Full MSE suite")