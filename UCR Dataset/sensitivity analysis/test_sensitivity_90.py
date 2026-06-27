import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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

class CNN_TemporalPooling(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
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
LENGTH = 90
ABLATE_REGION = (75, 90)  # this model's own last 15 timesteps


def train_full_model(model_class, seed, length, epochs=50, patience=10, lr=1e-3):
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

    model = model_class(NUM_CLASSES).to(device)
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


def ablate_region(X, start, end, mode='flatten'):
    X_ablated = X.copy()
    if mode == 'flatten':
        anchor = X[:, start-1:start]
        X_ablated[:, start:end] = anchor
    return X_ablated


SEEDS = [0, 42, 123, 256, 789]
results = []

for seed in SEEDS:
    model_pool, X_test_at_len = train_full_model(CNN_TemporalPooling, seed, LENGTH)
    model_lstm, _ = train_full_model(CNN_LSTM, seed, LENGTH)

    X_orig_t = torch.tensor(X_test_at_len).unsqueeze(1).to(device)
    with torch.no_grad():
        acc_pool_orig = accuracy_score(y_test_full, model_pool(X_orig_t).argmax(dim=1).cpu().numpy())
        acc_lstm_orig = accuracy_score(y_test_full, model_lstm(X_orig_t).argmax(dim=1).cpu().numpy())

    X_abl = ablate_region(X_test_at_len, *ABLATE_REGION)
    X_abl_t = torch.tensor(X_abl).unsqueeze(1).to(device)
    with torch.no_grad():
        acc_pool_abl = accuracy_score(y_test_full, model_pool(X_abl_t).argmax(dim=1).cpu().numpy())
        acc_lstm_abl = accuracy_score(y_test_full, model_lstm(X_abl_t).argmax(dim=1).cpu().numpy())

    pool_drop = acc_pool_orig - acc_pool_abl
    lstm_drop = acc_lstm_orig - acc_lstm_abl

    print(f"Seed {seed} | Pool: {acc_pool_orig:.4f} -> {acc_pool_abl:.4f} (drop: {pool_drop:+.4f}) | "
          f"LSTM: {acc_lstm_orig:.4f} -> {acc_lstm_abl:.4f} (drop: {lstm_drop:+.4f})")

    results.append({'seed': seed, 'pool_drop': pool_drop, 'lstm_drop': lstm_drop})

df = pd.DataFrame(results)
df.to_csv(f'ablation_T{LENGTH}_lastregion.csv', index=False)

print(f"\n{'='*65}")
print(f"T={LENGTH} — ablating own last 15 timesteps (t={ABLATE_REGION[0]}-{ABLATE_REGION[1]}, LOW cross-class variance)")
print(f"{'='*65}")
print(f"CNN+Pooling | mean drop: {df['pool_drop'].mean():.4f} ± {df['pool_drop'].std():.4f}")
print(f"CNN+LSTM    | mean drop: {df['lstm_drop'].mean():.4f} ± {df['lstm_drop'].std():.4f}")
print(f"\nCompare directly against T=140 result: LSTM drop was 0.3381 ± 0.0135 there.")
print(f"If T=90's LSTM drop here is MUCH smaller -> confirms it's about CONTENT (variance),")
print(f"not just blind recency to whatever the last 15 steps happen to be.")
print(f"If T=90's drop is comparably large -> it's positional recency regardless of content.")