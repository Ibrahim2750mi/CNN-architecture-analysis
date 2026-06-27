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


SEEDS = [42, 123, 789]

def swap_regions(X, region_a, region_b):
    """Swap the values in two equal-length regions."""
    a_start, a_end = region_a
    b_start, b_end = region_b
    assert (a_end - a_start) == (b_end - b_start), "Regions must be same length"
    X_swapped = X.copy()
    X_swapped[:, a_start:a_end] = X[:, b_start:b_end]
    X_swapped[:, b_start:b_end] = X[:, a_start:a_end]
    return X_swapped


def swap_regions_with_original(X_test_full_original, X_test_at_len, region_a, region_b, length):
    """
    Replace 75-90 in the length-truncated sequence with 
    actual 125-140 content from the FULL 140-length sequence.
    """
    a_start, a_end = region_a  # 75-90 (target)
    b_start, b_end = region_b  # 125-140 (source from original full data)
    
    # X_test_full_original is shape (samples, 140) - the FULL sequence
    # X_test_at_len is shape (samples, length) - the truncated sequence
    
    X_swapped = X_test_at_len.copy()
    
    # ALWAYS replace 75-90 with content from original 125-140
    X_swapped[:, a_start:a_end] = X_test_full_original[:, b_start:b_end]

    if length >= 140:
        X_swapped[:, b_start:b_end] = X_test_full_original[:, a_start:a_end]
    
    return X_swapped


# REGION_LOWVAR = (75, 90)
# REGION_HIGHVAR = (125, 140)
# lengths = [90, 100, 110, 120, 140]

# results_swap = []

# for length in lengths:
#     print(f"\nModel trained on length T={length}")
#     for seed in SEEDS:
#         model_pool, X_test_at_len = train_full_model(CNN_TemporalPooling, seed, length)
#         model_lstm, _ = train_full_model(CNN_LSTM, seed, length)

#         X_orig_t = torch.tensor(X_test_at_len).unsqueeze(1).to(device)
#         with torch.no_grad():
#             acc_pool_orig = accuracy_score(y_test_full, model_pool(X_orig_t).argmax(dim=1).cpu().numpy())
#             acc_lstm_orig = accuracy_score(y_test_full, model_lstm(X_orig_t).argmax(dim=1).cpu().numpy())

#         X_swapped = swap_regions_with_original(
#             X_test_full,  # Full 140-length original
#             X_test_at_len,    # Truncated version (length varies)
#             REGION_LOWVAR, 
#             REGION_HIGHVAR,
#             length
#         )
#         X_swap_t = torch.tensor(X_swapped).unsqueeze(1).to(device)
#         with torch.no_grad():
#             acc_pool_swap = accuracy_score(y_test_full, model_pool(X_swap_t).argmax(dim=1).cpu().numpy())
#             acc_lstm_swap = accuracy_score(y_test_full, model_lstm(X_swap_t).argmax(dim=1).cpu().numpy())

#         print(f"Seed {seed} | Pool: {acc_pool_orig:.4f} -> {acc_pool_swap:.4f} (Δ{acc_pool_swap-acc_pool_orig:+.4f}) | "
#               f"LSTM: {acc_lstm_orig:.4f} -> {acc_lstm_swap:.4f} (Δ{acc_lstm_swap-acc_lstm_orig:+.4f})")

#         results_swap.append({
#             'seed': seed,
#             'pool_orig': acc_pool_orig, 'pool_swap': acc_pool_swap,
#             'lstm_orig': acc_lstm_orig, 'lstm_swap': acc_lstm_swap,
#             'length': length
#         })

# df_swap = pd.DataFrame(results_swap)
# df_swap.to_csv('swap_test_T140.csv', index=False)

# print(f"\n{'='*60}\nSWAP TEST SUMMARY (T=140, swap t=75-90 <-> t=125-140)\n{'='*60}")
# print(f"LSTM | mean change: {(df_swap['lstm_swap']-df_swap['lstm_orig']).mean():+.4f} ± {(df_swap['lstm_swap']-df_swap['lstm_orig']).std():.4f}")
# print(f"Pool | mean change: {(df_swap['pool_swap']-df_swap['pool_orig']).mean():+.4f} ± {(df_swap['pool_swap']-df_swap['pool_orig']).std():.4f}")
# print(f"\nIf LSTM accuracy drops sharply (high-variance content moved AWAY from position 125-140)")
# print(f"despite the high-variance content still existing in the sequence (just relocated) ->")
# print(f"position is what matters, content moving elsewhere doesn't help.")


def extract_window(X, start, end):
    return X[:, start:end]


WINDOW_HIGHVAR = (120, 140)
WINDOW_LOWVAR  = (75, 95)

results_window = []

for seed in SEEDS:
    # Train fresh, small models using ONLY each 20-step window as input
    set_seed(seed)
    X_norm = normalize_per_sample(X_train_full)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_norm, y_train_full, test_size=0.15, stratify=y_train_full, random_state=seed)
    X_test_norm = normalize_per_sample(X_test_full)

    for label, (start, end) in [('highvar_120_140', WINDOW_HIGHVAR), ('lowvar_75_95', WINDOW_LOWVAR)]:
        X_tr_win  = extract_window(X_tr, start, end)
        X_val_win = extract_window(X_val, start, end)
        X_test_win = extract_window(X_test_norm, start, end)

        X_tr_t  = torch.tensor(X_tr_win).unsqueeze(1)
        X_val_t = torch.tensor(X_val_win).unsqueeze(1)
        y_tr_t, y_val_t = torch.tensor(y_tr), torch.tensor(y_val)

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)

        for model_name, model_class in [('Pool', CNN_TemporalPooling), ('LSTM', CNN_LSTM)]:
            set_seed(seed)
            model = model_class(NUM_CLASSES).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            best_val_acc, patience_counter, best_state = 0.0, 0, None

            for epoch in range(50):
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
                    if patience_counter >= 10: break

            model.load_state_dict(best_state)
            model.eval()
            X_test_t = torch.tensor(X_test_win).unsqueeze(1).to(device)
            with torch.no_grad():
                preds_test = model(X_test_t).argmax(dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test_full, preds_test)

            print(f"Seed {seed} | {model_name} | window={label} | test_acc={test_acc:.4f}")
            results_window.append({'seed': seed, 'model': model_name, 'window': label, 'test_acc': test_acc})

df_window = pd.DataFrame(results_window)
df_window.to_csv('window_headtohead_T20.csv', index=False)

print(f"\n{'='*60}\nHEAD-TO-HEAD: t=120-140 vs t=75-95, AS STANDALONE 20-STEP INPUTS\n{'='*60}")
print(df_window.groupby(['model', 'window'])['test_acc'].agg(['mean', 'std']))

