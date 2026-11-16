import os
import ctypes
from xml.parsers.expat import model

# Windows workaround: load the system C++ runtime before PyTorch.
if os.name == "nt":
    cpp_runtime = ctypes.WinDLL(
        os.path.join(os.environ["SystemRoot"], "System32", "msvcp140.dll")
    )

import copy
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "conv1d_v2"
REPORT_DIR = ROOT / "reports" / "conv1d_v2"

WINDOW = 200
STRIDE = 100
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
SEED = 42

# Participants never overlap between splits.
TRAIN_USERS = list(range(1, 10))
VAL_USERS = [10, 11]
TEST_USERS = [12, 13]

FEATURES = [
    f"{location}_{measurement}"
    for location in ["trousers", "back", "hand"]
    for measurement in [
        "Ax_g", "Ay_g", "Az_g",
        "Gx_dps", "Gy_dps", "Gz_dps",
    ]
]


class Autoencoder(nn.Module):
    """1D convolutional autoencoder with a 32-value latent vector."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Input after transpose: (batch, 18, 200)
            nn.Conv1d(len(FEATURES), 32, kernel_size=5,
                      stride=2, padding=2),
            nn.ReLU(),                         # (batch, 32, 100)

            nn.Conv1d(32, 64, kernel_size=5,
                      stride=2, padding=2),
            nn.ReLU(),                         # (batch, 64, 50)

            nn.Conv1d(64, 64, kernel_size=5,
                      stride=2, padding=2),
            nn.ReLU(),                         # (batch, 64, 25)

            nn.Flatten(),
            nn.Linear(64 * 25, 32),             # 32 latent values
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64 * 25),
            nn.ReLU(),
            nn.Unflatten(1, (64, 25)),

            nn.ConvTranspose1d(64, 64, kernel_size=4,
                               stride=2, padding=1),
            nn.ReLU(),                         # (batch, 64, 50)

            nn.ConvTranspose1d(64, 32, kernel_size=4,
                               stride=2, padding=1),
            nn.ReLU(),                         # (batch, 32, 100)

            nn.ConvTranspose1d(32, len(FEATURES), kernel_size=4,
                               stride=2, padding=1),
            # Output: (batch, 18, 200)
        )

    def forward(self, x):
        # Dataset format: (batch, time, channels)
        x = x.transpose(1, 2)
        reconstructed = self.decoder(self.encoder(x))
        return reconstructed.transpose(1, 2)


def load_recordings():
    files = sorted(DATA_DIR.rglob("activity_*_user_*_combined.csv"))
    if not files:
        raise FileNotFoundError(f"No sensor CSV files found in {DATA_DIR}")

    records = []
    audit = []
    seen = set()

    for path in files:
        match = re.fullmatch(
            r"activity_(\d+)_user_(\d+)_combined.csv", path.name
        )
        if not match:
            continue

        activity, user = map(int, match.groups())
        if (activity, user) in seen:
            raise ValueError(f"Duplicate recording: {path.name}")
        seen.add((activity, user))

        df = pd.read_csv(path)
        missing = set(FEATURES + ["datetime"]) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")

        values = df[FEATURES].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)

        times = pd.to_timedelta(
            df["datetime"], errors="coerce"
        ).dt.total_seconds().to_numpy()

        dt = np.diff(times)
        regular = (
            len(dt) > 0
            and np.isfinite(times).all()
            and np.allclose(dt, 0.01, atol=0.0001, rtol=0)
        )

        audit.append({
            "file": path.name,
            "user": user,
            "activity": activity,
            "rows": len(values),
            "invalid_values": int((~np.isfinite(values)).sum()),
            "regular_100hz": bool(regular),
        })

        # Stop rather than silently bridge gaps or use the wrong timing.
        if not regular:
            pd.DataFrame(audit).to_csv(
                REPORT_DIR / "data_audit.csv", index=False
            )
            raise ValueError(
                f"Unexpected timestamps in {path.name}. "
                "Inspect reports/data_audit.csv before training."
            )

        records.append((user, activity, path.name, values))

    pd.DataFrame(audit).to_csv(
        REPORT_DIR / "data_audit.csv", index=False
    )
    print(f"Loaded {len(records)} recordings.", flush=True)
    return records


def make_windows(records, users, mean, scale, stride):
    windows, metadata = [], []

    for user, activity, name, values in records:
        if user not in users:
            continue

        for start in range(0, len(values) - WINDOW + 1, stride):
            window = values[start:start + WINDOW]

            # Skip invalid windows without joining disconnected samples.
            if not np.isfinite(window).all():
                continue

            windows.append((window - mean) / scale)
            metadata.append({
                "file": name,
                "user": user,
                "activity": activity,
                "start_sample": start,
            })

    if not windows:
        raise ValueError(f"No valid windows for participants {users}")

    return np.stack(windows).astype(np.float32), metadata


def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            loss = (model(x) - x).square().mean(dim=(1, 2))
            total += loss.sum().item()
            count += len(x)

    return total / count


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)

    records = load_recordings()

    # Fit normalization ONLY on training participants.
    training_rows = np.concatenate([
        values for user, _, _, values in records if user in TRAIN_USERS
    ])
    training_rows = training_rows[
        np.isfinite(training_rows).all(axis=1)
    ]
    if not len(training_rows):
        raise ValueError("No finite training rows.")

    mean = training_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = training_rows.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    del training_rows

    train_x, _ = make_windows(
        records, TRAIN_USERS, mean, scale, STRIDE
    )
    val_x, _ = make_windows(
        records, VAL_USERS, mean, scale, WINDOW
    )
    test_x, test_meta = make_windows(
        records, TEST_USERS, mean, scale, WINDOW
    )

    print(
        f"Windows — train: {len(train_x)}, "
        f"validation: {len(val_x)}, test: {len(test_x)}",
        flush=True,
    )

    def loader(array, shuffle=False):
        return DataLoader(
            TensorDataset(torch.from_numpy(array)),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = loader(train_x, shuffle=True)
    val_loader = loader(val_x)
    test_loader = loader(test_x)

    model = Autoencoder().to(device)
    print("Model:", type(model))
    print("Parameter count:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - x).square().mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)

        train_loss = total / len(train_x)
        val_loss = evaluate(model, val_loader, device)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Non-finite training loss.")

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss,
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train MSE: {train_loss:.5f} | "
            f"validation MSE: {val_loss:.5f}",
            flush=True,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Early stopping.", flush=True)
                break

    model.load_state_dict(best_state)
    test_loss = evaluate(model, test_loader, device)

    # Baseline: reconstruct every value using the training channel mean.
    baseline_loss = float(np.mean(test_x ** 2))

    config = {
        "model": "Conv1dAutoencoder",
        "window": WINDOW,
        "training_stride": STRIDE,
        "evaluation_stride": WINDOW,
        "sampling_rate_hz": 100,
        "latent_dim": 32,
        "features": FEATURES,
        "train_users": TRAIN_USERS,
        "validation_users": VAL_USERS,
        "test_users": TEST_USERS,
        "seed": SEED,
        "torch_version": str(torch.__version__),
    }

    torch.save({
        "model_state_dict": {
            k: v.cpu() for k, v in model.state_dict().items()
        },
        "config": config,
    }, MODEL_DIR / "autoencoder.pt")

    np.savez(
        MODEL_DIR / "scaler.npz",
        mean=mean,
        scale=scale,
    )
    (MODEL_DIR / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "test_mse": test_loss,
        "training_mean_baseline_test_mse": baseline_loss,
        "train_windows": len(train_x),
        "validation_windows": len(val_x),
        "test_windows": len(test_x),
        "evaluation_split" : "test",
        "evaluation_users" : TEST_USERS,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(REPORT_DIR / "training_history.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(history_df.epoch, history_df.train_mse, label="Training")
    plt.plot(
        history_df.epoch, history_df.validation_mse, label="Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE (standardized signals)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "loss_curve.png", dpi=160)
    plt.close()

    model.eval()
    errors = []
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            errors.extend(
                (model(x) - x).square().mean(dim=(1, 2)).cpu().tolist()
            )

        # Fixed example, not selected for looking especially good.
        reconstruction = model(
            torch.from_numpy(test_x[:1]).to(device)
        ).cpu().numpy()[0]

    results = pd.DataFrame(test_meta)
    results["reconstruction_mse"] = errors
    results.to_csv(REPORT_DIR / "test_window_errors.csv", index=False)
    results.groupby("activity")["reconstruction_mse"].agg(
        ["count", "mean", "median"]
    ).to_csv(REPORT_DIR / "test_errors_by_activity.csv")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    seconds = np.arange(WINDOW) / 100
    for ax, channel in zip(axes, [0, 6, 12]):
        ax.plot(
            seconds, test_x[0, :, channel], label="Original", linewidth=1
        )
        ax.plot(
            seconds, reconstruction[:, channel],
            label="Reconstructed", linewidth=1
        )
        ax.set_ylabel(FEATURES[channel] + "\n(standardized)")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Test reconstruction — {test_meta[0]['file']}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "reconstruction.png", dpi=160)
    plt.close(fig)

    print("\nTraining complete.", flush=True)
    print(f"Test MSE: {test_loss:.5f}")
    print(f"Training-mean baseline MSE: {baseline_loss:.5f}")
    print(f"Model saved in: {MODEL_DIR}")
    print(f"Results saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()