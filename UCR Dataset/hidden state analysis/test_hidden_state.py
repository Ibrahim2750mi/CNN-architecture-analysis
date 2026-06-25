import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Model definitions — must match training exactly
# ============================================================

class CNN_LSTM(nn.Module):
    """Original model — used to load the saved checkpoints."""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128,
                            num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


class CNN_LSTM_Probe(nn.Module):
    """Same architecture, but forward() also returns the full
    hidden-state trajectory (layer-2 output at every timestep),
    not just the final h_n used for classification."""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128,
                            num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_states=False):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        all_h, (h_n, c_n) = self.lstm(x)  # all_h: (batch, T, 128) — every timestep's layer-2 output
        out = self.classifier(h_n[-1])
        if return_states:
            return out, all_h
        return out


def load_weights_from(probe_model, trained_model):
    probe_model.cnn.load_state_dict(trained_model.cnn.state_dict())
    probe_model.lstm.load_state_dict(trained_model.lstm.state_dict())
    probe_model.classifier.load_state_dict(trained_model.classifier.state_dict())
    return probe_model


def get_hidden_trajectory(model, X_sample, device):
    """X_sample: (N, 1, T). Returns ||h_t|| at every timestep, shape (N, T)."""
    probe = CNN_LSTM_Probe(model.classifier.out_features).to(device)
    probe = load_weights_from(probe, model)
    probe.eval()
    with torch.no_grad():
        _, all_h = probe(X_sample.to(device), return_states=True)
        norms = all_h.norm(dim=2)  # (N, T)
    return norms.cpu().numpy()


# ============================================================
# 2. Rebuild ECG5000 test data (same preprocessing as training)
# ============================================================

if not os.path.exists("ECG5000_TRAIN.txt"):
    os.system('wget -q "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"')
    os.system('unzip -q ECG5000.zip')

train_df = pd.read_csv("ECG5000_TRAIN.txt", header=None, sep=r'\s+')
test_df  = pd.read_csv("ECG5000_TEST.txt",  header=None, sep=r'\s+')

X_train_full = train_df.iloc[:, 1:].values.astype(np.float32)
y_train_full = train_df.iloc[:, 0].values
X_test_full  = test_df.iloc[:, 1:].values.astype(np.float32)
y_test_full  = test_df.iloc[:, 0].values

le = LabelEncoder()
y_train_full = le.fit_transform(y_train_full).astype(np.int64)
y_test_full  = le.transform(y_test_full).astype(np.int64)

def normalize_per_sample(X):
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

X_test_norm = normalize_per_sample(X_test_full)
X_test_t = torch.tensor(X_test_norm).unsqueeze(1)  # (N, 1, 140)
y_test_t = torch.tensor(y_test_full)

NUM_CLASSES = len(np.unique(y_train_full))
print(f"Test set: {X_test_t.shape}, Classes: {NUM_CLASSES}")


# ============================================================
# 3. Load trained models for T=40 and T=140
# ============================================================

SEED_FOLDER = "Seed 42"  # change to "seed 123", "seed 789", etc.

model_lstm_T40 = CNN_LSTM(NUM_CLASSES).to(device)
model_lstm_T40.load_state_dict(torch.load(f"{SEED_FOLDER}/best_lstm_40.pt", map_location=device))
model_lstm_T40.eval()

model_lstm_T140 = CNN_LSTM(NUM_CLASSES).to(device)
model_lstm_T140.load_state_dict(torch.load(f"{SEED_FOLDER}/best_lstm_140.pt", map_location=device))
model_lstm_T140.eval()

print(f"Loaded checkpoints from '{SEED_FOLDER}'")


# ============================================================
# 4. Build matching test inputs at each length
# ============================================================

X_test_T40  = X_test_t[:, :, :40]
X_test_T140 = X_test_t[:, :, :140]


# ============================================================
# 5. Run the warmup comparison
# ============================================================

def plot_warmup_comparison(model_T40, model_T140, X_test_T40, X_test_T140, n_samples=50):
    norms_40  = get_hidden_trajectory(model_T40,  X_test_T40[:n_samples],  device)   # (n_samples, 40)
    norms_140 = get_hidden_trajectory(model_T140, X_test_T140[:n_samples], device)   # (n_samples, 140)

    mean_40  = norms_40.mean(axis=0)
    std_40   = norms_40.std(axis=0)
    mean_140 = norms_140.mean(axis=0)
    std_140  = norms_140.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(mean_40, label='T=40 (mean ||h_t||)')
    axes[0].fill_between(range(len(mean_40)), mean_40 - std_40, mean_40 + std_40, alpha=0.2)
    axes[0].set_title('Hidden State Norm Trajectory — T=40')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('||h_t||')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(mean_140, label='T=140 (mean ||h_t||)', color='orange')
    axes[1].fill_between(range(len(mean_140)), mean_140 - std_140, mean_140 + std_140, alpha=0.2, color='orange')
    axes[1].set_title('Hidden State Norm Trajectory — T=140')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('||h_t||')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('warmup_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    def settling_score(norms):
        T = norms.shape[1]
        first_half = norms[:, :T // 2].std(axis=1).mean()
        second_half = norms[:, T // 2:].std(axis=1).mean()
        return first_half, second_half

    fh40, sh40 = settling_score(norms_40)
    fh140, sh140 = settling_score(norms_140)

    print(f"\nT=40  | First-half timestep-to-timestep variability: {fh40:.4f} | Second-half: {sh40:.4f}")
    print(f"T=140 | First-half timestep-to-timestep variability: {fh140:.4f} | Second-half: {sh140:.4f}")
    print(f"\nIf the warmup hypothesis is correct: T=140's second-half variability should be")
    print(f"noticeably LOWER than its first-half (settled), while T=40 may not show this drop")
    print(f"because the sequence ends before settling occurs.")

    np.savez('warmup_norms.npz', norms_40=norms_40, norms_140=norms_140)
    print("\nSaved raw per-sample, per-timestep norms to warmup_norms.npz")

    return norms_40, norms_140


norms_40, norms_140 = plot_warmup_comparison(
    model_lstm_T40, model_lstm_T140, X_test_T40, X_test_T140, n_samples=50
)