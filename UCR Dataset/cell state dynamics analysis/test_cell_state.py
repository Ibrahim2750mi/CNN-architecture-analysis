import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Model + probe that exposes BOTH h_t and c_t at every step
#    (your earlier probe only captured h_t via all_h; cell state
#    requires manually unrolling the LSTM cell-by-cell)
# ============================================================

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


class CNN_LSTM_CellProbe(nn.Module):
    """Manually unrolls the 2-layer LSTM using LSTMCell so we can
    capture c_t (cell state) at EVERY timestep, not just h_t."""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.cell1 = nn.LSTMCell(128, 128)
        self.cell2 = nn.LSTMCell(128, 128)
        self.classifier = nn.Linear(128, num_classes)

    def load_from_lstm(self, lstm_module):
        """Copy weights from nn.LSTM (2-layer) into the two LSTMCells."""
        # nn.LSTM stores weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 for layer 0
        # and weight_ih_l1 etc. for layer 1
        sd = lstm_module.state_dict()
        self.cell1.weight_ih.data.copy_(sd['weight_ih_l0'])
        self.cell1.weight_hh.data.copy_(sd['weight_hh_l0'])
        self.cell1.bias_ih.data.copy_(sd['bias_ih_l0'])
        self.cell1.bias_hh.data.copy_(sd['bias_hh_l0'])

        self.cell2.weight_ih.data.copy_(sd['weight_ih_l1'])
        self.cell2.weight_hh.data.copy_(sd['weight_hh_l1'])
        self.cell2.bias_ih.data.copy_(sd['bias_ih_l1'])
        self.cell2.bias_hh.data.copy_(sd['bias_hh_l1'])

    def forward(self, x, return_cell_states=False):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, T, 128)
        B, T, _ = x.shape

        h1 = torch.zeros(B, 128, device=x.device)
        c1 = torch.zeros(B, 128, device=x.device)
        h2 = torch.zeros(B, 128, device=x.device)
        c2 = torch.zeros(B, 128, device=x.device)

        all_c2 = []  # cell state of LAYER 2 at every timestep (the one feeding the classifier)
        all_h2 = []

        for t in range(T):
            xt = x[:, t, :]
            h1, c1 = self.cell1(xt, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            all_c2.append(c2)
            all_h2.append(h2)

        all_c2 = torch.stack(all_c2, dim=1)  # (B, T, 128)
        all_h2 = torch.stack(all_h2, dim=1)  # (B, T, 128)

        out = self.classifier(h2)  # final timestep's h2, matches original model's h_n[-1]

        if return_cell_states:
            return out, all_h2, all_c2
        return out


def load_weights(probe, trained_model):
    probe.cnn.load_state_dict(trained_model.cnn.state_dict())
    probe.load_from_lstm(trained_model.lstm)
    probe.classifier.load_state_dict(trained_model.classifier.state_dict())
    return probe


def get_cell_state_trajectory(model, X_sample, device):
    """Returns c_t values at every timestep: shape (N, T, 128)."""
    probe = CNN_LSTM_CellProbe(model.classifier.out_features).to(device)
    probe = load_weights(probe, model)
    probe.eval()
    with torch.no_grad():
        _, all_h, all_c = probe(X_sample.to(device), return_cell_states=True)
    return all_h.cpu().numpy(), all_c.cpu().numpy()


# ============================================================
# 2. Sanity check: verify the unrolled probe matches the
#    original nn.LSTM-based model's output exactly
# ============================================================

def sanity_check(model, probe_model, X_sample, device):
    model.eval()
    with torch.no_grad():
        out_original = model(X_sample.to(device))
    out_probe, _, _ = probe_model(X_sample.to(device), return_cell_states=True)
    max_diff = (out_original - out_probe).abs().max().item()
    print(f"Sanity check — max output difference between original and probe: {max_diff:.8f}")
    print("(should be ~1e-5 or smaller; if large, weight loading is wrong)")


# ============================================================
# 3. Rebuild ECG5000 test data
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
X_test_t = torch.tensor(X_test_norm).unsqueeze(1)
NUM_CLASSES = len(np.unique(y_train_full))

X_test_T40  = X_test_t[:, :, :40]
X_test_T140 = X_test_t[:, :, :140]


# ============================================================
# 4. Run across seeds, collect cell-state magnitude near end
# ============================================================

SEEDS = ["seed 42", "seed 123", "seed 789"]
N_SAMPLES = 100
LAST_K = 5  # look at the last 5 timesteps before sequence ends

c40_last_all, c140_last_all = [], []

for i, seed_folder in enumerate(SEEDS):
    model_T40 = CNN_LSTM(NUM_CLASSES).to(device)
    model_T40.load_state_dict(torch.load(f"{seed_folder}/best_lstm_40.pt", map_location=device))
    model_T40.eval()

    model_T140 = CNN_LSTM(NUM_CLASSES).to(device)
    model_T140.load_state_dict(torch.load(f"{seed_folder}/best_lstm_140.pt", map_location=device))
    model_T140.eval()

    if i == 0:
        # Run sanity check once
        probe_check = CNN_LSTM_CellProbe(NUM_CLASSES).to(device)
        probe_check = load_weights(probe_check, model_T40)
        sanity_check(model_T40, probe_check, X_test_T40[:10], device)

    h40, c40 = get_cell_state_trajectory(model_T40, X_test_T40[:N_SAMPLES], device)
    h140, c140 = get_cell_state_trajectory(model_T140, X_test_T140[:N_SAMPLES], device)

    # c40/c140 shape: (N_SAMPLES, T, 128) -- take the last LAST_K timesteps, all 128 dims, all samples
    c40_last  = c40[:, -LAST_K:, :].flatten()    # last 5 steps before T=40 cutoff
    c140_last = c140[:, -LAST_K:, :].flatten()   # last 5 steps before T=140 cutoff

    c40_last_all.append(c40_last)
    c140_last_all.append(c140_last)

    frac_near_zero_40  = (np.abs(c40_last)  < 0.5).mean()
    frac_near_zero_140 = (np.abs(c140_last) < 0.5).mean()
    print(f"{seed_folder} | T=40 frac |c_t|<0.5 (last {LAST_K} steps): {frac_near_zero_40:.3f} | "
          f"T=140: {frac_near_zero_140:.3f}")

c40_combined  = np.concatenate(c40_last_all)
c140_combined = np.concatenate(c140_last_all)


# ============================================================
# 5. Plot the distribution of |c_t| near sequence end
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(np.abs(c40_combined), bins=60, alpha=0.6, label='T=40', color='steelblue', density=True)
axes[0].hist(np.abs(c140_combined), bins=60, alpha=0.6, label='T=140', color='orange', density=True)
axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='|c_t|=0.5 (high tanh-sensitivity zone)')
axes[0].set_xlabel('|c_t| in last 5 timesteps')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of |cell state| near sequence end')
axes[0].legend()
axes[0].set_xlim(0, 5)

# tanh derivative as a function of c_t, for reference
c_range = np.linspace(0, 5, 200)
tanh_deriv = 1 - np.tanh(c_range)**2
axes[1].plot(c_range, tanh_deriv, color='purple')
axes[1].set_xlabel('|c_t|')
axes[1].set_ylabel("tanh'(c_t) = 1 - tanh(c_t)^2")
axes[1].set_title("tanh sensitivity curve — where the magic happens")
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cellstate_tanh_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nOverall (3 seeds combined, {LAST_K} last steps, 128 dims, {N_SAMPLES} samples):")
print(f"T=40  | mean |c_t|: {np.abs(c40_combined).mean():.4f} | frac <0.5: {(np.abs(c40_combined)<0.5).mean():.4f}")
print(f"T=140 | mean |c_t|: {np.abs(c140_combined).mean():.4f} | frac <0.5: {(np.abs(c140_combined)<0.5).mean():.4f}")