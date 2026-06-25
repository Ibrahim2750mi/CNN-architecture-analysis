import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. Load ECG5000 (raw, unnormalized — we want to see real waveform shape)
# ============================================================

if not os.path.exists("ECG5000_TRAIN.txt"):
    os.system('wget -q "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"')
    os.system('unzip -q ECG5000.zip')

train_df = pd.read_csv("ECG5000_TRAIN.txt", header=None, sep=r'\s+')
test_df  = pd.read_csv("ECG5000_TEST.txt",  header=None, sep=r'\s+')

X_train = train_df.iloc[:, 1:].values.astype(np.float32)
y_train = train_df.iloc[:, 0].values
X_test  = test_df.iloc[:, 1:].values.astype(np.float32)
y_test  = test_df.iloc[:, 0].values

le = LabelEncoder()
y_train = le.fit_transform(y_train).astype(np.int64)
y_test  = le.transform(y_test).astype(np.int64)

print(f"Test set: {X_test.shape}, Classes: {np.unique(y_test)}")
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

T = X_test_norm.shape[1]  # 140
timesteps = np.arange(T)

overall_mean = X_test_norm.mean(axis=0)
overall_std  = X_test_norm.std(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# --- Panel 1: overall average waveform with T=40 marked ---
axes[0].plot(timesteps, overall_mean, color='black', linewidth=2, label='Mean waveform (all classes)')
axes[0].fill_between(timesteps, overall_mean - overall_std, overall_mean + overall_std,
                      alpha=0.15, color='gray', label='±1 std')
axes[0].axvline(x=40, color='red', linestyle='--', linewidth=2, label='T=40 cutoff')
axes[0].axvline(x=30, color='orange', linestyle=':', linewidth=1.5, label='T=30 cutoff')
axes[0].axvline(x=50, color='green', linestyle=':', linewidth=1.5, label='T=50 cutoff')
axes[0].set_title('ECG5000 — Average Normalized Waveform (Test Set, All Classes)')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Normalized Amplitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Panel 2: per-class average waveform, same markers ---
classes = np.unique(y_test)
colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

for cls, color in zip(classes, colors):
    class_mean = X_test_norm[y_test == cls].mean(axis=0)
    n_samples = (y_test == cls).sum()
    axes[1].plot(timesteps, class_mean, label=f'Class {cls} (n={n_samples})', color=color)

axes[1].axvline(x=40, color='red', linestyle='--', linewidth=2, label='T=40 cutoff')
axes[1].axvline(x=30, color='orange', linestyle=':', linewidth=1.5, label='T=30 cutoff')
axes[1].axvline(x=50, color='green', linestyle=':', linewidth=1.5, label='T=50 cutoff')
axes[1].set_title('ECG5000 — Average Waveform by Class')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Normalized Amplitude')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ecg5000_waveform_t40_check.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 4. Quantify: rate of change (slope) near each cutoff point
#    — large slope at a cutoff = truncating mid-feature
# ============================================================

def local_slope(signal, idx, window=3):
    """Approximate derivative at idx using a small window."""
    lo = max(0, idx - window)
    hi = min(len(signal), idx + window)
    return (signal[hi-1] - signal[lo]) / (hi - lo)

print("\n=== Slope of average waveform at each candidate cutoff ===")
for cutoff in [20, 30, 40, 50, 70, 100, 140]:
    if cutoff < T:
        slope = local_slope(overall_mean, cutoff)
        print(f"T={cutoff:<5} | local slope of mean waveform: {slope:+.4f}")

print("\nA cutoff landing where |slope| is large means it's truncating mid-feature")
print("(e.g. during a rising/falling edge like a QRS complex), rather than at a flat,")
print("quiescent part of the waveform.")


# ============================================================
# 5. Show individual sample examples (not just the average) —
#    averaging can hide real per-sample structure if beats aren't aligned
# ============================================================

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
np.random.seed(0)
sample_idxs = np.random.choice(len(X_test_norm), 3, replace=False)

for ax, idx in zip(axes2, sample_idxs):
    ax.plot(timesteps, X_test_norm[idx], color='steelblue')
    ax.axvline(x=40, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=30, color='orange', linestyle=':', linewidth=1)
    ax.axvline(x=50, color='green', linestyle=':', linewidth=1)
    ax.set_title(f'Sample {idx} (class {y_test[idx]})')
    ax.grid(True, alpha=0.3)

axes2[-1].set_xlabel('Timestep')
plt.tight_layout()
plt.savefig('ecg5000_individual_samples_t40_check.png', dpi=150, bbox_inches='tight')
plt.show()