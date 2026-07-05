import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. Load FordA (raw, unnormalized — we want to see real waveform shape)
# ============================================================
# NOTE: FordA labels are {-1, 1} (not 1-5 like ECG5000), T=500 (not 140).

if not os.path.exists("FordA_TRAIN.txt"):
    os.system('wget -q "https://www.timeseriesclassification.com/aeon-toolkit/FordA.zip"')
    os.system('unzip -q FordA.zip')

train_df = pd.read_csv("FordA_TRAIN.txt", header=None, sep=r'\s+')
test_df  = pd.read_csv("FordA_TEST.txt",  header=None, sep=r'\s+')

X_train = train_df.iloc[:, 1:].values.astype(np.float32)
y_train = train_df.iloc[:, 0].values
X_test  = test_df.iloc[:, 1:].values.astype(np.float32)
y_test  = test_df.iloc[:, 0].values

le = LabelEncoder()
y_train = le.fit_transform(y_train).astype(np.int64)
y_test  = le.transform(y_test).astype(np.int64)

print(f"Test set: {X_test.shape}, Classes (encoded): {np.unique(y_test)}")
print(f"Class distribution (test): {np.bincount(y_test)}")


# ============================================================
# 2. Normalize per-sample (same as training pipeline) so shape
#    comparison is on the same scale models actually see
# ============================================================

def normalize_per_sample(X):
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

X_test_norm = normalize_per_sample(X_test)


# ============================================================
# 3. Plot average waveform — overall, and per class
# ============================================================

T = X_test_norm.shape[1]  # 500 for FordA
timesteps = np.arange(T)

overall_mean = X_test_norm.mean(axis=0)
overall_std  = X_test_norm.std(axis=0)

# Candidate cutoffs spread across the full 500-length sequence.
# (FordA has no established trough location yet — these are exploratory
# markers spaced across the range, unlike ECG5000 where 30/40/50 were
# already known to matter.)
candidate_cutoffs = [50, 100, 150, 250, 350, 450]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# --- Panel 1: overall average waveform with candidate cutoffs marked ---
axes[0].plot(timesteps, overall_mean, color='black', linewidth=1.5, label='Mean waveform (all classes)')
axes[0].fill_between(timesteps, overall_mean - overall_std, overall_mean + overall_std,
                      alpha=0.15, color='gray', label='±1 std')
cutoff_colors = plt.cm.autumn(np.linspace(0, 0.8, len(candidate_cutoffs)))
for cutoff, color in zip(candidate_cutoffs, cutoff_colors):
    axes[0].axvline(x=cutoff, color=color, linestyle='--', linewidth=1.2, label=f'T={cutoff}')
axes[0].set_title('FordA — Average Normalized Waveform (Test Set, All Classes)')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Normalized Amplitude')
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(True, alpha=0.3)

# --- Panel 2: per-class average waveform, same markers ---
classes = np.unique(y_test)
colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

for cls, color in zip(classes, colors):
    class_mean = X_test_norm[y_test == cls].mean(axis=0)
    n_samples = (y_test == cls).sum()
    axes[1].plot(timesteps, class_mean, label=f'Class {cls} (n={n_samples})', color=color)

for cutoff, color in zip(candidate_cutoffs, cutoff_colors):
    axes[1].axvline(x=cutoff, color=color, linestyle='--', linewidth=1.2)
axes[1].set_title('FordA — Average Waveform by Class')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Normalized Amplitude')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forda_waveform_check.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 4. Cross-class variance profile — where does discriminative
#    signal actually live along the 500 timesteps? (mirrors the
#    t=125-140 high-variance finding from ECG5000, Finding 3)
# ============================================================

class_means = np.stack([X_test_norm[y_test == cls].mean(axis=0) for cls in classes])
cross_class_variance = class_means.var(axis=0)  # variance across classes, per timestep

fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(timesteps, cross_class_variance, color='darkred', linewidth=1.2)
for cutoff, color in zip(candidate_cutoffs, cutoff_colors):
    ax3.axvline(x=cutoff, color=color, linestyle='--', linewidth=1)
ax3.set_title('FordA — Cross-Class Variance by Timestep (where discriminative signal lives)')
ax3.set_xlabel('Timestep')
ax3.set_ylabel('Variance across class means')
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forda_cross_class_variance.png', dpi=150, bbox_inches='tight')
plt.show()

top_variance_region = np.argsort(cross_class_variance)[-15:]
print(f"\nTop 15 highest cross-class-variance timesteps: {sorted(top_variance_region.tolist())}")
print(f"Mean variance in last 15 timesteps (t=485-500): {cross_class_variance[-15:].mean():.5f}")
print(f"Mean variance overall: {cross_class_variance.mean():.5f}")


# ============================================================
# 5. Rate of change (slope) near each cutoff point
#    — large slope at a cutoff = truncating mid-feature
# ============================================================

def local_slope(signal, idx, window=3):
    """Approximate derivative at idx using a small window."""
    lo = max(0, idx - window)
    hi = min(len(signal), idx + window)
    return (signal[hi-1] - signal[lo]) / (hi - lo)

print("\n=== Slope of average waveform at each candidate cutoff ===")
for cutoff in candidate_cutoffs + [T]:
    if cutoff <= T:
        idx = min(cutoff, T - 1)
        slope = local_slope(overall_mean, idx)
        print(f"T={cutoff:<5} | local slope of mean waveform: {slope:+.4f}")

print("\nA cutoff landing where |slope| is large means it's truncating mid-feature")
print("rather than at a flat, quiescent part of the waveform.")


# ============================================================
# 6. Local activity (mean |slope| over trailing window before
#    each candidate cutoff) — this is the FordA analogue of the
#    'mean activity 0.142 at T=40' local-minimum finding on ECG5000
# ============================================================

def trailing_activity(signal, cutoff, window=5):
    lo = max(0, cutoff - window)
    diffs = np.abs(np.diff(signal[lo:cutoff]))
    return diffs.mean() if len(diffs) > 0 else np.nan

print("\n=== Trailing local activity (mean |slope| over last 5 steps before cutoff) ===")
for cutoff in candidate_cutoffs + [T]:
    idx = min(cutoff, T)
    activity = trailing_activity(overall_mean, idx)
    print(f"T={cutoff:<5} | trailing activity: {activity:.4f}")


# ============================================================
# 7. Show individual sample examples (not just the average) —
#    averaging can hide real per-sample structure if signals
#    aren't phase-aligned (FordA is engine noise — likely NOT
#    aligned the way ECG beats roughly are)
# ============================================================

fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
np.random.seed(0)
sample_idxs = np.random.choice(len(X_test_norm), 3, replace=False)

for ax, idx in zip(axes2, sample_idxs):
    ax.plot(timesteps, X_test_norm[idx], color='steelblue', linewidth=0.8)
    for cutoff, color in zip(candidate_cutoffs, cutoff_colors):
        ax.axvline(x=cutoff, color=color, linestyle='--', linewidth=1)
    ax.set_title(f'Sample {idx} (class {y_test[idx]})')
    ax.grid(True, alpha=0.3)

axes2[-1].set_xlabel('Timestep')
plt.tight_layout()
plt.savefig('forda_individual_samples.png', dpi=150, bbox_inches='tight')
plt.show()
