"""Simple entry point. Default: Part 1. Example: python run_lab.py --task vae --device cuda"""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--task", choices=["part1", "faces", "cifar", "vae", "unet", "gan", "audit"], default="part1")
    args, remaining = parser.parse_known_args()
    task = args.task
    module = "mri" if task in ["vae", "unet", "gan"] else "data" if task == "audit" else task
    sys.argv = [sys.argv[0], *([task] if module == "mri" else []), *remaining]
    importlib.import_module("lab." + module).main()


if __name__ == "__main__":
    main()
