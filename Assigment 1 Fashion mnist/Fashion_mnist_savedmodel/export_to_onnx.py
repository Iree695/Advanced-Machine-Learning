import subprocess
import sys
import os

SAVED_MODEL_DIR = "Fashion_mnist_savedmodel"
ONNX_MODEL_PATH = "fashion_mnist_cnn.onnx"

def main():
    if not os.path.isdir(SAVED_MODEL_DIR):
        raise FileNotFoundError(
            f"No existe la carpeta '{SAVED_MODEL_DIR}'. "
            f"Ejecuta primero Training.py."
        )

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", SAVED_MODEL_DIR,
        "--output", ONNX_MODEL_PATH,
        "--opset", "13"
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    print("ONNX creado:", ONNX_MODEL_PATH)

if __name__ == "__main__":
    main()
