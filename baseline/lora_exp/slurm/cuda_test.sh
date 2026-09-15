#!/usr/bin/env bash

#SBATCH -J cuda-test
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=8G
#SBATCH --time=00:10:00
#SBATCH -o runs/cuda-test-%j.out
#SBATCH -e runs/cuda-test-%j.err

set -euo pipefail

PROJECT_DIR="/data/${USER}/repos/lab/kv_cache/baseline/lora_exp"
cd "${PROJECT_DIR}"

source /data/"${USER}"/anaconda3/etc/profile.d/conda.sh
conda activate lab
unset LD_LIBRARY_PATH

echo "================================================"
echo "[SYSTEM]"
echo "date=$(date)"
echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID:-unset}"
echo "================================================"

echo
echo "[SLURM / CUDA ENV]"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

echo
echo "[NVIDIA-SMI]"
nvidia-smi

echo
echo "[NVIDIA DEVICE FILES]"
ls -l /dev/nvidia* || true

echo
echo "[PYTHON]"
which python
python --version

echo
echo "================================================"
echo "[CUDA DRIVER API DIRECT TEST]"
echo "================================================"

echo "[DRIVER KERNEL VERSION]"
cat /proc/driver/nvidia/version || true

echo
echo "[LIBCUDA]"
ldconfig -p | grep -E 'libcuda\.so' || true

LIBCUDA_PATH="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')"
echo "resolved libcuda.so.1=${LIBCUDA_PATH:-<not found>}"

if [ -n "${LIBCUDA_PATH:-}" ]; then
    ls -l "$LIBCUDA_PATH" || true
    readlink -f "$LIBCUDA_PATH" || true
fi

echo
echo "[CGROUP]"
cat /proc/self/cgroup || true

echo
echo "[DIRECT cuInit()]"

python - <<'PY'
import ctypes
import ctypes.util

print("ctypes.find_library('cuda') =", ctypes.util.find_library("cuda"))

lib = ctypes.CDLL("libcuda.so.1")
print("CDLL(libcuda.so.1)          = PASS")

lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int

lib.cuGetErrorName.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
]
lib.cuGetErrorName.restype = ctypes.c_int

lib.cuGetErrorString.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
]
lib.cuGetErrorString.restype = ctypes.c_int

ret = lib.cuInit(0)

name = ctypes.c_char_p()
desc = ctypes.c_char_p()

lib.cuGetErrorName(ret, ctypes.byref(name))
lib.cuGetErrorString(ret, ctypes.byref(desc))

print("cuInit return code =", ret)
print("cuInit error name  =", name.value.decode() if name.value else "<none>")
print("cuInit description =", desc.value.decode() if desc.value else "<none>")

driver_ver = ctypes.c_int()
drv_ret = lib.cuDriverGetVersion(ctypes.byref(driver_ver))

print("cuDriverGetVersion return =", drv_ret)
print("CUDA driver API version   =", driver_ver.value)
PY

echo
echo "================================================"
echo "[SLURM DEVICE CGROUP CHECK]"
echo "================================================"

CGROUP_DEV="/sys/fs/cgroup/devices/slurm/uid_${UID}/job_${SLURM_JOB_ID}/step_batch/task_0"

echo "CGROUP_DEV=$CGROUP_DEV"

if [ -f "${CGROUP_DEV}/devices.list" ]; then
    echo
    echo "[devices.list]"
    cat "${CGROUP_DEV}/devices.list"

    echo
    echo "[NVIDIA-related entries]"
    grep -E '195:|511:|509:|506:|507:' "${CGROUP_DEV}/devices.list" || true
else
    echo "devices.list not found"
fi

echo
echo "[DIRECT DEVICE OPEN TEST]"

python - <<'PY'
import os

devices = [
    "/dev/nvidiactl",
    "/dev/nvidia0",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
]

for dev in devices:
    try:
        fd = os.open(dev, os.O_RDWR)
        print(f"{dev:25s} OPEN PASS")
        os.close(fd)
    except Exception as e:
        print(f"{dev:25s} OPEN FAIL: {type(e).__name__}: {e}")
PY

echo
echo "[PYTORCH / CUDA]"
python - <<'PY'
import os
import sys
import torch

print("python executable :", sys.executable)
print("torch file        :", torch.__file__)
print("torch version     :", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)

print("CUDA_VISIBLE_DEVICES:",
      os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

try:
    print("cuda available    :", torch.cuda.is_available())
except Exception as e:
    print("cuda available EXCEPTION:", repr(e))

try:
    print("cuda device count :", torch.cuda.device_count())
except Exception as e:
    print("device count EXCEPTION:", repr(e))

try:
    if torch.cuda.is_available():
        print("current device    :", torch.cuda.current_device())
        print("device name       :", torch.cuda.get_device_name(0))

        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()

        print("CUDA tensor test  : PASS")
        print("tensor device     :", y.device)
    else:
        print("CUDA tensor test  : SKIPPED")
except Exception as e:
    print("CUDA tensor test EXCEPTION:", repr(e))

print("================================================")
print(torch.__config__.show())
PY