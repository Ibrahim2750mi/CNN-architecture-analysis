import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Re-define model classes (must match training exactly)
# ============================================================

class CNN_TemporalPooling_Forecast(nn.Module):
    def __init__(self, pred_len=96):
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
        self.forecaster = nn.Linear(128, pred_len)

    def forward(self, x):
        x = self.network(x)
        x = self.gap(x).squeeze(-1)
        return self.forecaster(x)


class CNN_LSTM_Forecast(nn.Module):
    def __init__(self, pred_len=96):
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
        self.forecaster = nn.Linear(128, pred_len)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.forecaster(h_n[-1])


# ============================================================
# 2. Rebuild test data EXACTLY as in original ETTh1 pipeline
#    (must match preprocessing used when models were trained)
# ============================================================

df = pd.read_csv('ETTh1.csv')
df = df.drop(columns=['date'])
values = df['OT'].values.reshape(-1, 1)

n = len(values)
train_end = int(n * 0.6)
val_end   = int(n * 0.8)

train_data = values[:train_end]
test_data  = values[val_end:]

mean = train_data.mean()
std  = train_data.std()
test_data_norm = (test_data - mean) / std

def make_sequences(data, input_len=336, pred_len=96):
    X, y = [], []
    for i in range(len(data) - input_len - pred_len):
        X.append(data[i : i+input_len])
        y.append(data[i+input_len : i+input_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_test, y_test = make_sequences(test_data_norm)
X_test_t = torch.tensor(X_test).permute(0, 2, 1)  # (N, 1, 336)

print(f"Test set shape: {X_test_t.shape}")


# ============================================================
# 3. Sensitivity analysis function
# ============================================================

def compute_input_sensitivity(model, X_sample, device, n_perturb_trials=20, noise_std=0.5):
    model.eval()
    T = X_sample.shape[2]
    sensitivities = np.zeros(T)

    X_sample = X_sample.to(device)
    with torch.no_grad():
        baseline_output = model(X_sample).cpu().numpy()

    for t in range(T):
        diffs = []
        for _ in range(n_perturb_trials):
            X_perturbed = X_sample.clone()
            noise = torch.randn(1).item() * noise_std
            X_perturbed[0, 0, t] += noise

            with torch.no_grad():
                perturbed_output = model(X_perturbed).cpu().numpy()

            diff = np.abs(perturbed_output - baseline_output).mean()
            diffs.append(diff)

        sensitivities[t] = np.mean(diffs)

    return sensitivities


# ============================================================
# 4. Run sensitivity test per seed, then average across seeds
# ============================================================

SEEDS = [42, 123, 789]
N_SAMPLES = 10
T = 336

all_pool_sens = []
all_lstm_sens = []

for seed in SEEDS:
    print(f"\n{'='*50}\nSEED {seed}\n{'='*50}")

    pool_path = f"{seed}/best_model_pooling_{seed}.pt"
    lstm_path = f"{seed}/best_model_lstm_{seed}.pt"

    if not (os.path.exists(pool_path) and os.path.exists(lstm_path)):
        print(f"Model files not found for seed {seed}, skipping. Checked: {pool_path}, {lstm_path}")
        continue

    pool_model = CNN_TemporalPooling_Forecast().to(device)
    pool_model.load_state_dict(torch.load(pool_path, map_location=device))
    pool_model.eval()

    lstm_model = CNN_LSTM_Forecast().to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    pool_sens_seed = np.zeros((N_SAMPLES, T))
    lstm_sens_seed = np.zeros((N_SAMPLES, T))

    for i in range(N_SAMPLES):
        X_sample = X_test_t[i:i+1]
        pool_sens_seed[i] = compute_input_sensitivity(pool_model, X_sample, device)
        lstm_sens_seed[i] = compute_input_sensitivity(lstm_model, X_sample, device)
        print(f"  Sample {i+1}/{N_SAMPLES} done")

    all_pool_sens.append(pool_sens_seed.mean(axis=0))
    all_lstm_sens.append(lstm_sens_seed.mean(axis=0))

# Average across seeds
pool_mean_sens = np.mean(all_pool_sens, axis=0)
lstm_mean_sens = np.mean(all_lstm_sens, axis=0)

pool_mean_sens_norm = pool_mean_sens / pool_mean_sens.sum()
lstm_mean_sens_norm = lstm_mean_sens / lstm_mean_sens.sum()


# ============================================================
# 5. Plot
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(pool_mean_sens_norm, label='CNN+Pooling', alpha=0.8)
plt.plot(lstm_mean_sens_norm, label='CNN+LSTM', alpha=0.8)
plt.xlabel('Input Timestep (0 = 336 hours before prediction, 335 = most recent)')
plt.ylabel('Normalized Sensitivity')
plt.title('Which input timesteps does each model actually rely on? (averaged over 3 seeds)')
plt.legend()
plt.grid(True)
plt.savefig('sensitivity_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 6. Quantitative recency bias score
# ============================================================

recent_window = slice(306, 336)   # last 30 hours
distant_window = slice(0, 100)    # first 100 hours

pool_recent = pool_mean_sens_norm[recent_window].sum()
pool_distant = pool_mean_sens_norm[distant_window].sum()
lstm_recent = lstm_mean_sens_norm[recent_window].sum()
lstm_distant = lstm_mean_sens_norm[distant_window].sum()

print(f"\n{'='*50}\nRECENCY BIAS RESULTS (averaged over {len(all_pool_sens)} seeds)\n{'='*50}")
print(f"CNN+Pooling | Recent (last 30h): {pool_recent:.4f} | Distant (first 100h): {pool_distant:.4f} | Ratio: {pool_recent/pool_distant:.2f}")
print(f"CNN+LSTM    | Recent (last 30h): {lstm_recent:.4f} | Distant (first 100h): {lstm_distant:.4f} | Ratio: {lstm_recent/lstm_distant:.2f}")

with open('sensitivity_results.pkl', 'wb') as f:
    pickle.dump({
        'pool_mean_sens': pool_mean_sens,
        'lstm_mean_sens': lstm_mean_sens,
        'pool_mean_sens_norm': pool_mean_sens_norm,
        'lstm_mean_sens_norm': lstm_mean_sens_norm,
    }, f)
print("\nSaved results to sensitivity_results.pkl")