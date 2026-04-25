import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


class ModelStore:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir).resolve()
        self.model_path = None
        self.model_mtime = None
        self.weights = None
        self.model_name = ""

    def latest_model_path(self):
        model_paths = [
            path for path in self.models_dir.glob("*.npz")
            if path.is_file() and not path.name.endswith("_training_history.npz")
        ]
        if not model_paths:
            raise FileNotFoundError(f"No .npz model found in {self.models_dir}")
        return max(model_paths, key=lambda path: path.stat().st_mtime)

    def load_latest(self):
        latest_path = self.latest_model_path()
        latest_mtime = latest_path.stat().st_mtime
        if latest_path == self.model_path and latest_mtime == self.model_mtime:
            return

        data = np.load(latest_path)
        self.weights = {
            "W1": data["W1"],
            "b1": data["b1"],
            "W2": data["W2"],
            "b2": data["b2"],
        }
        raw_name = data["model_name"] if "model_name" in data else latest_path.stem
        self.model_name = str(raw_name.item() if hasattr(raw_name, "item") else raw_name)
        self.model_path = latest_path
        self.model_mtime = latest_mtime

    def info(self):
        self.load_latest()
        return {
            "modelName": self.model_name,
            "modelPath": str(self.model_path.relative_to(PROJECT_ROOT)),
            "hiddenSize": int(self.weights["W1"].shape[0]),
        }

    def predict(self, pixels):
        self.load_latest()
        x = np.asarray(pixels, dtype=np.float64).reshape(1, 784)
        x = np.clip(x, 0.0, 1.0)

        z1 = x @ self.weights["W1"].T + self.weights["b1"]
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.weights["W2"].T + self.weights["b2"]
        probabilities = softmax(z2)[0]
        prediction = int(np.argmax(probabilities))

        return {
            "prediction": prediction,
            "confidence": float(probabilities[prediction]),
            "probabilities": [float(value) for value in probabilities],
            **self.info(),
        }


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def build_handler(model_store):
    class MNISTHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def do_GET(self):
            if urlparse(self.path).path == "/api/model":
                self.write_json(model_store.info())
                return
            super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path != "/api/predict":
                self.send_error(404, "Endpoint not found")
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length) or b"{}")
                pixels = payload.get("pixels")
                if not isinstance(pixels, list) or len(pixels) != 784:
                    raise ValueError("Request body must contain a pixels array with 784 numbers")
                self.write_json(model_store.predict(pixels))
            except Exception as exc:
                self.write_json({"error": str(exc)}, status=400)

        def write_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return MNISTHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Run MNIST drawing web app")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Directory containing saved .npz models")
    return parser.parse_args()


def main():
    args = parse_args()
    if not WEB_ROOT.exists():
        raise FileNotFoundError(f"Web assets not found: {WEB_ROOT}")

    model_store = ModelStore(args.models_dir)
    model_info = model_store.info()
    handler = build_handler(model_store)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Loaded model: {model_info['modelName']} ({model_info['modelPath']})")
    print(f"Open http://{args.host}:{args.port} in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
