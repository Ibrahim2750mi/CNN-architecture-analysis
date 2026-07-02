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
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

X_test_norm = normalize_per_sample(X_test)

# ============================================================
# Measure signal activity immediately BEFORE each cutoff
# ============================================================

def recent_activity_before_cutoff(X, cutoff, window=5):
    """
    Computes the total absolute movement over the last `window`
    timesteps before the cutoff.

    Activity = |Σ x[t+1] - x[t]|

    Returns one value per sample.
    """
    start = max(0, cutoff - window)

    segment = X[:, start:cutoff]          # (N, window)

    # Consecutive differences
    diffs = np.diff(segment, axis=1)

    # Total absolute movement
    activity = np.abs(diffs.sum(axis=1))

    return activity


print("="*70)
print("Signal activity during the LAST 5 timesteps before each cutoff")
print("(Higher = signal changing rapidly)")
print("(Lower = flatter / less informative region)")
print("="*70)

results = []

for cutoff in [20, 30, 40, 45, 50, 70, 100, 140]:

    activity = recent_activity_before_cutoff(
        X_test_norm,
        cutoff=cutoff,
        window=10
    )

    mean_activity = activity.mean()
    std_activity = activity.std()

    results.append({
        "Cutoff": cutoff,
        "MeanActivity": mean_activity,
        "StdActivity": std_activity
    })

    print(
        f"T={cutoff:<4} | "
        f"Mean activity = {mean_activity:.4f} ± {std_activity:.4f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv("activity_before_cutoff.csv", index=False)

print("\nSaved results to activity_before_cutoff.csv")

lowest = results_df.loc[results_df["MeanActivity"].idxmin()]

print("\nLeast active region:")
print(
    f"T={int(lowest.Cutoff)} "
    f"(Mean activity = {lowest.MeanActivity:.4f})"
)

print("\nInterpretation:")
print("If T=40 has the LOWEST activity, it supports the hypothesis")
print("that the LSTM is being truncated during a relatively flat")
print("and uninformative portion of the ECG waveform.")