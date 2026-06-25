import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

if not os.path.exists("ECG5000_TRAIN.txt"):
    os.system('wget -q "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"')
    os.system('unzip -q ECG5000.zip')

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

# ============================================================
# For each candidate cutoff, look at the slope over the
# LAST 5 timesteps BEFORE the cutoff (not the slope AT the
# cutoff point itself)
# ============================================================

def recent_slope_before_cutoff(X, cutoff, window=5):
    """For each sample, compute how much the signal changed over
    the `window` timesteps right before `cutoff`. Returns array of
    shape (N,) -- one value per sample."""
    start = max(0, cutoff - window)
    segment = X[:, start:cutoff]  # (N, window)
    # slope = (last value - first value) in this window
    slope = segment[:, -1] - segment[:, 0]
    return slope

print("=== Average |change| over the last 5 timesteps BEFORE each cutoff ===")
print("(Large value = signal was actively moving right before truncation)")
print("(Small value = signal was flat/boring right before truncation)\n")

for cutoff in [20, 30, 40, 50, 70, 100, 140]:
    slopes = recent_slope_before_cutoff(X_test_norm, cutoff, window=5)
    mean_abs_change = np.abs(slopes).mean()
    std_abs_change = np.abs(slopes).std()
    print(f"T={cutoff:<5} | mean |change| in last 5 steps: {mean_abs_change:.4f} (std: {std_abs_change:.4f})")

print("\nIf T=40 has a noticeably SMALLER value here than T=30 and T=50,")
print("that supports: 'LSTM gets cut off during a flat/uninformative stretch at T=40'")