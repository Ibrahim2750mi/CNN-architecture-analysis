import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Model definitions
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
# 2. Rebuild test set
# ============================================================

if not os.path.exists('ETTh1.csv'):
    os.system('wget -q "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"')

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
y_test_arr = y_test.squeeze(-1) if y_test.ndim == 3 else y_test  # (N, 96)

print(f"Test set: X={X_test_t.shape}, y={y_test_arr.shape}")

# Persistence baseline (same for every seed — doesn't depend on the model)
last_values = X_test_t[:, 0, -1].numpy()
persistence_preds = np.tile(last_values.reshape(-1, 1), (1, y_test_arr.shape[1]))
persistence_mse = mean_squared_error(y_test_arr.flatten(), persistence_preds.flatten())
persistence_mae = mean_absolute_error(y_test_arr.flatten(), persistence_preds.flatten())

print(f"\nPersistence Baseline | MSE: {persistence_mse:.4f} | MAE: {persistence_mae:.4f}")


# ============================================================
# 3. Helper to get predictions
# ============================================================

def get_predictions(model, X, batch_size=32):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(device)
            pred = model(batch).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


# ============================================================
# 4. Loop over all 3 seeds
# ============================================================

SEEDS = [42, 123, 789]
results = []

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    pool_path = f"{seed}/best_model_pooling_{seed}.pt"
    lstm_path = f"{seed}/best_model_lstm_{seed}.pt"

    if not (os.path.exists(pool_path) and os.path.exists(lstm_path)):
        print(f"Checkpoint missing for seed {seed} — checked {pool_path}, {lstm_path}. Skipping.")
        continue

    pool_model = CNN_TemporalPooling_Forecast().to(device)
    pool_model.load_state_dict(torch.load(pool_path, map_location=device))
    pool_model.eval()

    lstm_model = CNN_LSTM_Forecast().to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    lstm_preds = get_predictions(lstm_model, X_test_t)
    pool_preds = get_predictions(pool_model, X_test_t)

    lstm_mse = mean_squared_error(y_test_arr.flatten(), lstm_preds.flatten())
    lstm_mae = mean_absolute_error(y_test_arr.flatten(), lstm_preds.flatten())
    pool_mse = mean_squared_error(y_test_arr.flatten(), pool_preds.flatten())
    pool_mae = mean_absolute_error(y_test_arr.flatten(), pool_preds.flatten())

    lstm_corr = np.corrcoef(lstm_preds.flatten(), persistence_preds.flatten())[0, 1]
    pool_corr = np.corrcoef(pool_preds.flatten(), persistence_preds.flatten())[0, 1]

    lstm_improve = (persistence_mse - lstm_mse) / persistence_mse * 100
    pool_improve = (persistence_mse - pool_mse) / persistence_mse * 100

    print(f"CNN+LSTM    | MSE: {lstm_mse:.4f} | MAE: {lstm_mae:.4f} | Corr w/ persistence: {lstm_corr:.4f} | Improvement: {lstm_improve:+.1f}%")
    print(f"CNN+Pooling | MSE: {pool_mse:.4f} | MAE: {pool_mae:.4f} | Corr w/ persistence: {pool_corr:.4f} | Improvement: {pool_improve:+.1f}%")

    results.append({
        'seed': seed,
        'lstm_mse': lstm_mse, 'lstm_mae': lstm_mae, 'lstm_corr': lstm_corr, 'lstm_improve': lstm_improve,
        'pool_mse': pool_mse, 'pool_mae': pool_mae, 'pool_corr': pool_corr, 'pool_improve': pool_improve,
    })


# ============================================================
# 5. Averaged summary across seeds
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv('persistence_comparison_all_seeds.csv', index=False)

print(f"\n{'='*70}")
print(f"AVERAGED ACROSS {len(results_df)} SEEDS")
print(f"{'='*70}")
print(f"Persistence Baseline | MSE: {persistence_mse:.4f} | MAE: {persistence_mae:.4f}")
print(f"CNN+LSTM              | MSE: {results_df['lstm_mse'].mean():.4f} ± {results_df['lstm_mse'].std():.4f} "
      f"| Corr: {results_df['lstm_corr'].mean():.4f} | Improvement: {results_df['lstm_improve'].mean():+.1f}%")
print(f"CNN+Pooling            | MSE: {results_df['pool_mse'].mean():.4f} ± {results_df['pool_mse'].std():.4f} "
      f"| Corr: {results_df['pool_corr'].mean():.4f} | Improvement: {results_df['pool_improve'].mean():+.1f}%")
print(f"{'='*70}")

print(results_df)