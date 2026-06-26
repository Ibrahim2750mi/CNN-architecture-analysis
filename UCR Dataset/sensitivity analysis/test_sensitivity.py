import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


# ============================================================
# Load data + train CNN_LSTM at T=45, multiple seeds
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

NUM_CLASSES = len(np.unique(y_train_full))
LENGTH = 45
SEEDS = [42, 123, 789]

def train_at_length(seed, length, epochs=50, patience=10, lr=1e-3):
    set_seed(seed)
    X_norm = normalize_per_sample(X_train_full)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_norm, y_train_full, test_size=0.15, stratify=y_train_full, random_state=seed)
    X_test_norm = normalize_per_sample(X_test_full)

    X_tr_t  = torch.tensor(X_tr[:, :length]).unsqueeze(1)
    X_val_t = torch.tensor(X_val[:, :length]).unsqueeze(1)
    y_tr_t, y_val_t = torch.tensor(y_tr), torch.tensor(y_val)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

    model = CNN_LSTM(NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, patience_counter, best_state = 0.0, 0, None

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward(); optimizer.step()
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                p = model(Xb.to(device)).argmax(dim=1).cpu().numpy()
                preds.extend(p); labels.extend(yb.numpy())
        val_acc = accuracy_score(labels, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc; patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience: break

    model.load_state_dict(best_state)
    model.eval()
    return model, X_test_norm[:, :length]


# ============================================================
# STEP 1 — Direct sensitivity probe: which timesteps does
# the T=45 model actually rely on?
# ============================================================

def compute_input_sensitivity(model, X_sample, device, n_trials=20, noise_std=0.5):
    model.eval()
    T = X_sample.shape[2]
    sensitivities = np.zeros(T)
    X_sample = X_sample.to(device)
    with torch.no_grad():
        baseline = model(X_sample).cpu().numpy()
    for t in range(T):
        diffs = []
        for _ in range(n_trials):
            X_pert = X_sample.clone()
            noise = torch.randn(X_pert.shape[0], device=device) * noise_std
            X_pert[:, 0, t] += noise
            with torch.no_grad():
                pert_out = model(X_pert).cpu().numpy()
            diffs.append(np.abs(pert_out - baseline).mean())
        sensitivities[t] = np.mean(diffs)
    return sensitivities


sensitivity_results = []

for seed in SEEDS:
    model, X_test_at_len = train_at_length(seed, LENGTH)
    X_test_t = torch.tensor(X_test_at_len).unsqueeze(1)
    sens = compute_input_sensitivity(model, X_test_t[:30], device)  # 30 samples for speed
    sensitivity_results.append(sens)
    print(f"Seed {seed} | sensitivity at last 5 timesteps: {sens[-5:]}")

sens_arr = np.stack(sensitivity_results)
sens_mean, sens_std = sens_arr.mean(axis=0), sens_arr.std(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(sens_mean, marker='o', markersize=3)
plt.fill_between(range(LENGTH), sens_mean - sens_std, sens_mean + sens_std, alpha=0.2)
plt.xlabel('Timestep')
plt.ylabel('Sensitivity (output change under perturbation)')
plt.title(f'Which timesteps does LSTM at T={LENGTH} actually rely on? (mean ± std, 5 seeds)')
plt.grid(True, alpha=0.3)
plt.savefig(f'sensitivity_T{LENGTH}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nMost sensitive timestep: {sens_mean.argmax()} (value: {sens_mean.max():.4f})")
print(f"Sensitivity at very last timestep ({LENGTH-1}): {sens_mean[-1]:.4f}")


# ============================================================
# STEP 2 — Fabrication test: flatten last K timesteps of GOOD
# T=45 sequences, see if accuracy drops
# ============================================================

def flatten_tail(X, k):
    """Replace the last k timesteps with the value at position -(k+1),
    i.e. make the tail perfectly flat, removing any slope."""
    X_flat = X.copy()
    anchor_value = X[:, -(k+1)]  # value just before the flattened region
    for i in range(1, k+1):
        X_flat[:, -i] = anchor_value
    return X_flat


fabrication_results = []

for seed in SEEDS:
    model, X_test_at_len = train_at_length(seed, LENGTH)
    X_test_t_orig = torch.tensor(X_test_at_len).unsqueeze(1).to(device)
    y_test_t = torch.tensor(y_test_full)

    model.eval()
    with torch.no_grad():
        preds_orig = model(X_test_t_orig).argmax(dim=1).cpu().numpy()
    acc_orig = accuracy_score(y_test_full, preds_orig)

    row = {'seed': seed, 'k': 0, 'accuracy': acc_orig}
    fabrication_results.append(row)

    for k in [1, 3, 5, 8, 10]:
        X_flat = flatten_tail(X_test_at_len, k)
        X_flat_t = torch.tensor(X_flat).unsqueeze(1).to(device)
        with torch.no_grad():
            preds_flat = model(X_flat_t).argmax(dim=1).cpu().numpy()
        acc_flat = accuracy_score(y_test_full, preds_flat)
        fabrication_results.append({'seed': seed, 'k': k, 'accuracy': acc_flat})

    print(f"Seed {seed} | orig acc: {acc_orig:.4f} | " +
          " | ".join([f"k={r['k']}: {r['accuracy']:.4f}" for r in fabrication_results if r['seed']==seed and r['k']>0]))

fab_df = pd.DataFrame(fabrication_results)
fab_df.to_csv(f'fabrication_test_T{LENGTH}.csv', index=False)

summary = fab_df.groupby('k')['accuracy'].agg(['mean', 'std'])
print(f"\n{'='*50}\nFABRICATION TEST SUMMARY (T={LENGTH})\n{'='*50}")
print(summary)

plt.figure(figsize=(8, 5))
plt.errorbar(summary.index, summary['mean'], yerr=summary['std'], marker='o', capsize=4)
plt.xlabel('Number of flattened timesteps at the end (k)')
plt.ylabel('Test Accuracy')
plt.title(f'Does flattening the tail of T={LENGTH} sequences hurt accuracy?')
plt.grid(True, alpha=0.3)
plt.savefig(f'fabrication_test_T{LENGTH}.png', dpi=150, bbox_inches='tight')
plt.show()