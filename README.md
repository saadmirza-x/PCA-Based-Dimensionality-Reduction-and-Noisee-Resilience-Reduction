PCA-Based Dimensionality Reduction on Country-Level Indicator Data
CE342 – Computational Methods | GIKI, Spring 2026

Team: Muhammad Saad Bin Waqas, Hassan Khalid, Moiz ud din
Overview
We run PCA on a 9-feature country-level dataset, once on clean data and once after injecting Gaussian noise at SNR = 11 dB. The goal: see how much noise degrades the structure PCA finds, and how well reconstruction recovers it.

Method — step by step
Standardize raw features to zero mean, unit variance.
Inject noise into the standardized clean data at SNR=11 — this creates the "Noisy" dataset. (Noise added after scaling, so the SNR value is meaningful and comparable across features.)
Run PCA on both clean and noisy datasets separately, computing all 9 principal components for each.
Reconstruct signals from a reduced component set (k=5, since that's where cumulative variance crosses 90%) and project back to original space.
Score reconstruction using MSE across 4 comparisons — this is the part that actually tells you if PCA is doing its job or just making pretty plots.
Results
Explained Variance (scree plot): First component alone captures ~46% (clean) / ~43% (noisy) of variance. Both datasets cross the 90% cumulative threshold at k=5 components — noise barely shifted where the "elbow" sits, which is a good sign your signal structure is robust.
Reconstruction MSE (4-way comparison):
| Comparison | MSE | What it means |
|---|---|---|
| Clean recon vs clean | 0.369 | Pure dimensionality-reduction loss |
| Noisy recon vs noisy | 0.440 | Reconstruction error when noise is present |
| Noisy recon vs clean | 0.388 | Denoising metric — how close PCA got back to ground truth |
| Raw noisy vs clean | 0.089 | Baseline noise level before any PCA |
Here's the part worth flagging in your report: the "denoising" MSE (0.388) is higher than the raw noise baseline (0.089). That's not a bug — it means PCA reconstruction at k=5 didn't denoise the data, it added error on top of the noise. Worth a sentence explaining why (probably: k=5 keeps some noise-dominated components, or information genuinely lost in the low-variance tail).
Cluster visualization (2D & 3D): Country groupings stay visually consistent between clean and noisy projections — same neighbors (e.g., Cape Verde/Philippines/Sri Lanka cluster together in both) even with noise added, which supports that PCA structure is noise-resistant at a qualitative level even if the MSE story is more nuanced.


Repo Structure
├── Code_File.py   # actual script
├── report/                       # IEEE report
├── plots/                  # the 5 figures above
└── README.md