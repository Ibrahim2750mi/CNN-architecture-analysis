import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Model definitions
# ============================================================

class CNN_TemporalPooling(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class CNN_LSTM(nn.Module):
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
    """Exposes the full hidden-state trajectory, not just final h_n."""
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
        all_h, (h_n, c_n) = self.lstm(x)
        out = self.classifier(h_n[-1])
        if return_states:
            return out, all_h
        return out


class CNN_Pooling_Probe(nn.Module):
    """Exposes pre-GAP activations at every pooled timestep."""
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_states=False):
        feat = self.network(x)
        pooled = self.gap(feat).squeeze(-1)
        out = self.classifier(pooled)
        if return_states:
            return out, feat.permute(0, 2, 1)
        return out


def load_lstm_weights(probe_model, trained_model):
    probe_model.cnn.load_state_dict(trained_model.cnn.state_dict())
    probe_model.lstm.load_state_dict(trained_model.lstm.state_dict())
    probe_model.classifier.load_state_dict(trained_model.classifier.state_dict())
    return probe_model


def load_pool_weights(probe_model, trained_model):
    probe_model.network.load_state_dict(trained_model.network.state_dict())
    probe_model.classifier.load_state_dict(trained_model.classifier.state_dict())
    return probe_model


def get_lstm_hidden_trajectory(model, X_sample, device):
    probe = CNN_LSTM_Probe(model.classifier.out_features).to(device)
    probe = load_lstm_weights(probe, model)
    probe.eval()
    with torch.no_grad():
        _, all_h = probe(X_sample.to(device), return_states=True)
        norms = all_h.norm(dim=2)
    return norms.cpu().numpy()


def get_pooling_activation_trajectory(model, X_sample, device):
    probe = CNN_Pooling_Probe(model.classifier.out_features).to(device)
    probe = load_pool_weights(probe, model)
    probe.eval()
    with torch.no_grad():
        _, feat = probe(X_sample.to(device), return_states=True)
        norms = feat.norm(dim=2)
    return norms.cpu().numpy()


# ============================================================
# 2. Rebuild ECG5000 test data
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

X_test_T40  = X_test_t[:, :, :40]
X_test_T140 = X_test_t[:, :, :140]


# ============================================================
# 3. Loop over all 3 seeds, collect trajectories
# ============================================================

SEEDS = ["seed 42", "seed 123", "seed 789"]
N_SAMPLES = 50

lstm_norms_40_all, lstm_norms_140_all = [], []
pool_norms_140_all = []

for seed_folder in SEEDS:
    print(f"\n--- {seed_folder} ---")

    # LSTM hidden state trajectories
    model_lstm_T40 = CNN_LSTM(NUM_CLASSES).to(device)
    model_lstm_T40.load_state_dict(torch.load(f"{seed_folder}/best_lstm_40.pt", map_location=device))
    model_lstm_T40.eval()

    model_lstm_T140 = CNN_LSTM(NUM_CLASSES).to(device)
    model_lstm_T140.load_state_dict(torch.load(f"{seed_folder}/best_lstm_140.pt", map_location=device))
    model_lstm_T140.eval()

    norms_40  = get_lstm_hidden_trajectory(model_lstm_T40,  X_test_T40[:N_SAMPLES],  device)
    norms_140 = get_lstm_hidden_trajectory(model_lstm_T140, X_test_T140[:N_SAMPLES], device)
    lstm_norms_40_all.append(norms_40.mean(axis=0))    # average over samples -> (40,)
    lstm_norms_140_all.append(norms_140.mean(axis=0))  # average over samples -> (140,)

    # Pooling pre-GAP activation trajectory (T=140 only, matches the original comparison)
    model_pool_T140 = CNN_TemporalPooling(NUM_CLASSES).to(device)
    model_pool_T140.load_state_dict(torch.load(f"{seed_folder}/best_pool_140.pt", map_location=device))
    model_pool_T140.eval()

    pool_norms = get_pooling_activation_trajectory(model_pool_T140, X_test_T140[:N_SAMPLES], device)
    pool_norms_140_all.append(pool_norms.mean(axis=0))  # average over samples -> (35,) due to 2x pooling

    print(f"  LSTM T=40  final mean ||h||: {norms_40.mean(axis=0)[-1]:.3f}")
    print(f"  LSTM T=140 final mean ||h||: {norms_140.mean(axis=0)[-1]:.3f}")
    print(f"  Pool T=140 final mean ||feat||: {pool_norms.mean(axis=0)[-1]:.3f}")

# Stack across seeds: shape (3, T)
lstm_40_arr  = np.stack(lstm_norms_40_all)   # (3, 40)
lstm_140_arr = np.stack(lstm_norms_140_all)  # (3, 140)
pool_140_arr = np.stack(pool_norms_140_all)  # (3, 35)

lstm_40_mean,  lstm_40_std  = lstm_40_arr.mean(axis=0),  lstm_40_arr.std(axis=0)
lstm_140_mean, lstm_140_std = lstm_140_arr.mean(axis=0), lstm_140_arr.std(axis=0)
pool_140_mean, pool_140_std = pool_140_arr.mean(axis=0), pool_140_arr.std(axis=0)

np.savez('trajectory_summary_all_seeds.npz',
         lstm_40_mean=lstm_40_mean, lstm_40_std=lstm_40_std,
         lstm_140_mean=lstm_140_mean, lstm_140_std=lstm_140_std,
         pool_140_mean=pool_140_mean, pool_140_std=pool_140_std)
print("\nSaved trajectory_summary_all_seeds.npz")


# ============================================================
# 4. Plot — LSTM hidden state, T=40 vs T=140, mean±std across seeds
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

t40 = np.arange(len(lstm_40_mean))
axes[0].plot(t40, lstm_40_mean, label='T=40 (mean across 3 seeds)', color='steelblue')
axes[0].fill_between(t40, lstm_40_mean - lstm_40_std, lstm_40_mean + lstm_40_std, alpha=0.25, color='steelblue')
axes[0].set_title('LSTM Hidden State Norm — T=40 (mean ± std over seeds)')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('||h_t||')
axes[0].grid(True)
axes[0].legend()

t140 = np.arange(len(lstm_140_mean))
axes[1].plot(t140, lstm_140_mean, label='T=140 (mean across 3 seeds)', color='orange')
axes[1].fill_between(t140, lstm_140_mean - lstm_140_std, lstm_140_mean + lstm_140_std, alpha=0.25, color='orange')
axes[1].set_title('LSTM Hidden State Norm — T=140 (mean ± std over seeds)')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('||h_t||')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig('lstm_warmup_trajectory_3seeds.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 5. Plot — Pooling activation norm vs LSTM hidden state norm
#    at T=140, both mean±std across seeds, same x-axis
# ============================================================

fig2, ax = plt.subplots(figsize=(10, 5))

# Rescale pooling's compressed timestep axis (35 points) back to original 140-timestep scale
pool_x_rescaled = np.linspace(0, 140, len(pool_140_mean))

ax2 = ax.twinx()  # different y-scale since LSTM norms (~1-8) and Pooling norms (~0-100) differ a lot

l1 = ax.plot(t140, lstm_140_mean, label='LSTM ||h_t|| (left axis)', color='orange')
ax.fill_between(t140, lstm_140_mean - lstm_140_std, lstm_140_mean + lstm_140_std, alpha=0.2, color='orange')

l2 = ax2.plot(pool_x_rescaled, pool_140_mean, label='Pooling ||feat|| (right axis)', color='green')
ax2.fill_between(pool_x_rescaled, pool_140_mean - pool_140_std, pool_140_mean + pool_140_std, alpha=0.2, color='green')

ax.axvline(x=105, color='red', linestyle='--', alpha=0.5, label='waveform feature region')

ax.set_xlabel('Original timestep')
ax.set_ylabel('LSTM ||h_t||', color='orange')
ax2.set_ylabel('Pooling ||feat||', color='green')
ax.set_title('LSTM vs CNN+Pooling sensitivity at T=140 (mean across 3 seeds)')

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_vs_pooling_3seeds.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 6. Settling score, averaged with std across seeds
# ============================================================

def settling_score(norms_2d):
    """norms_2d: (n_samples, T) for ONE seed. Returns (first_half_var, second_half_var)."""
    T = norms_2d.shape[1]
    first_half = norms_2d[:, :T // 2].std(axis=1).mean()
    second_half = norms_2d[:, T // 2:].std(axis=1).mean()
    return first_half, second_half

print(f"\n{'='*60}\nSETTLING SCORES (per seed, then mean ± std)\n{'='*60}")

# Need per-seed full norms again (not just the mean trajectory) for this
fh40_list, sh40_list, fh140_list, sh140_list = [], [], [], []

for seed_folder in SEEDS:
    model_lstm_T40 = CNN_LSTM(NUM_CLASSES).to(device)
    model_lstm_T40.load_state_dict(torch.load(f"{seed_folder}/best_lstm_40.pt", map_location=device))
    model_lstm_T40.eval()

    model_lstm_T140 = CNN_LSTM(NUM_CLASSES).to(device)
    model_lstm_T140.load_state_dict(torch.load(f"{seed_folder}/best_lstm_140.pt", map_location=device))
    model_lstm_T140.eval()

    norms_40  = get_lstm_hidden_trajectory(model_lstm_T40,  X_test_T40[:N_SAMPLES],  device)
    norms_140 = get_lstm_hidden_trajectory(model_lstm_T140, X_test_T140[:N_SAMPLES], device)

    fh40, sh40 = settling_score(norms_40)
    fh140, sh140 = settling_score(norms_140)

    fh40_list.append(fh40); sh40_list.append(sh40)
    fh140_list.append(fh140); sh140_list.append(sh140)

    print(f"{seed_folder} | T=40 first/second-half var: {fh40:.4f}/{sh40:.4f} | "
          f"T=140 first/second-half var: {fh140:.4f}/{sh140:.4f}")

print(f"\n{'='*60}\nMEAN ± STD ACROSS 3 SEEDS\n{'='*60}")
print(f"T=40  | First-half: {np.mean(fh40_list):.4f} ± {np.std(fh40_list):.4f} | "
      f"Second-half: {np.mean(sh40_list):.4f} ± {np.std(sh40_list):.4f}")
print(f"T=140 | First-half: {np.mean(fh140_list):.4f} ± {np.std(fh140_list):.4f} | "
      f"Second-half: {np.mean(sh140_list):.4f} ± {np.std(sh140_list):.4f}")