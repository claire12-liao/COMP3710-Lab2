#!/usr/bin/env bash
# Submit from the project root. No invented partition/account/module names.
# sbatch --partition=<your-GPU-partition> --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00 cluster/run_job.sh cifar
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the project root}"
if [[ -z "${LAB_PYTHON:-}" ]]; then
  echo 'Set LAB_PYTHON to your course GPU environment Python before sbatch (see docs/RANGPUR.md).'
  exit 1
fi
if [[ ! -x "$LAB_PYTHON" ]]; then
  echo 'LAB_PYTHON must be an executable Python path.'
  exit 1
fi
"$LAB_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "A GPU allocation and CUDA-enabled PyTorch are required"; print(torch.cuda.get_device_name(0))'
task="${1:-part1}"
shift || true
case "$task" in
  part1) "$LAB_PYTHON" -m lab.part1 --device cuda "$@" ;;
  faces) "$LAB_PYTHON" -m lab.faces --device cuda "$@" ;;
  cifar) "$LAB_PYTHON" -m lab.cifar --device cuda --require-slurm "$@" ;;
  cifar-demo) "$LAB_PYTHON" -m lab.cifar --mode demo --device cuda --require-slurm "$@" ;;
  vae|unet|gan) "$LAB_PYTHON" -m lab.mri "$task" --device cuda "$@" ;;
  *) echo 'Choose part1, faces, cifar, cifar-demo, vae, unet, or gan'; exit 2 ;;
esac
