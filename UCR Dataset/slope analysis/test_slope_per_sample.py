import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from scipy import stats

test_df = pd.read_csv("ECG5000_TEST.txt", header=None, sep=r'\s+')
X_test = test_df.iloc[:, 1:].values.astype(np.float32)
y_test = test_df.iloc[:, 0].values
le = LabelEncoder()
y_test = le.fit_transform(y_test).astype(np.int64)

def normalize_per_sample(X):
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

X_test_norm = normalize_per_sample(X_test)

def recent_slope_before_cutoff(X, cutoff, window=5):
    start = max(0, cutoff - window)
    segment = X[:, start:cutoff]
    slope = segment[:, -1] - segment[:, 0]
    return slope

slope_40 = np.abs(recent_slope_before_cutoff(X_test_norm, 40, window=5))
slope_30 = np.abs(recent_slope_before_cutoff(X_test_norm, 30, window=5))
slope_50 = np.abs(recent_slope_before_cutoff(X_test_norm, 50, window=5))

print("=== Per-sample distribution check (not just aggregate mean) ===")
print(f"T=30 | median: {np.median(slope_30):.4f} | 25th pct: {np.percentile(slope_30,25):.4f} | 75th pct: {np.percentile(slope_30,75):.4f}")
print(f"T=40 | median: {np.median(slope_40):.4f} | 25th pct: {np.percentile(slope_40,25):.4f} | 75th pct: {np.percentile(slope_40,75):.4f}")
print(f"T=50 | median: {np.median(slope_50):.4f} | 25th pct: {np.percentile(slope_50,25):.4f} | 75th pct: {np.percentile(slope_50,75):.4f}")

stat_30_40, p_30_40 = stats.wilcoxon(slope_30, slope_40)
stat_50_40, p_50_40 = stats.wilcoxon(slope_50, slope_40)

print(f"\nWilcoxon signed-rank test (paired, same samples):")
print(f"T=30 vs T=40: statistic={stat_30_40:.1f}, p-value={p_30_40:.6f}")
print(f"T=50 vs T=40: statistic={stat_50_40:.1f}, p-value={p_50_40:.6f}")
print(f"\n(p < 0.05 means the difference is statistically significant, not just noise)")

frac_lower_than_30 = (slope_40 < slope_30).mean()
frac_lower_than_50 = (slope_40 < slope_50).mean()
print(f"\nFraction of samples where T=40 slope < T=30 slope: {frac_lower_than_30:.3f}")
print(f"Fraction of samples where T=40 slope < T=50 slope: {frac_lower_than_50:.3f}")
print("(0.5 = no real difference, pure noise; further from 0.5 = real, consistent effect)")