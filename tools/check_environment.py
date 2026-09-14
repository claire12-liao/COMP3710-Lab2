"""Run from anywhere: python tools/check_environment.py"""
import importlib
import platform
import sys

print("Python executable:", sys.executable)
print("Python version:", platform.python_version())
failures = []
for name in ["numpy", "matplotlib", "sklearn", "PIL", "torch", "torchvision"]:
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "available"))
    except Exception as exc:
        failures.append(name)
        print(name, "ERROR:", str(exc))
if not failures:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU tensor check:", (torch.ones(2, device="cuda") * 3).cpu().tolist())
    else:
        print("CPU works. GPU timing and required Rangpur demo remain to be run on a GPU allocation.")
else:
    raise SystemExit("Fix missing packages in the Python executable printed above: " + ", ".join(failures))
