import importlib.util
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SOURCE = ROOT / "src" / "train_autoencoder.py"

# Load the same dense model class used for training.
spec = importlib.util.spec_from_file_location("dense_train", SOURCE)
dense_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dense_train)

config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
checkpoint = torch.load(
    MODEL_DIR / "autoencoder.pt",
    map_location="cpu",
    weights_only=False,
)

model = dense_train.Autoencoder()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class DenseWithEmbedding(torch.nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        embedding = self.autoencoder.encoder(x)
        reconstruction = self.autoencoder.decoder(embedding)
        return embedding, reconstruction


export_model = DenseWithEmbedding(model).eval()

# Input shape: batch, time steps, sensor channels.
example = torch.zeros(
    1, config["window"], len(config["features"]), dtype=torch.float32
)
onnx_path = MODEL_DIR / "model.onnx"

torch.onnx.export(
    export_model,
    (example,),
    str(onnx_path),
    input_names=["sensor_window"],
    output_names=["embedding", "reconstruction"],
    dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
    external_data=False,
)

# Check that ONNX and PyTorch produce the same reconstruction.
session = ort.InferenceSession(
    str(onnx_path), providers=["CPUExecutionProvider"]
)

with torch.no_grad():
    expected_embedding = model.encoder(example).numpy()
    expected_reconstruction = model.decoder(
        model.encoder(example)
    ).numpy()

actual_embedding, actual_reconstruction = session.run(
    ["embedding", "reconstruction"],
    {"sensor_window": example.numpy()},
)

print(
    "Embedding matches:",
    np.allclose(expected_embedding, actual_embedding, atol=1e-5),
)
print(
    "Reconstruction matches:",
    np.allclose(expected_reconstruction, actual_reconstruction, atol=1e-5),
)