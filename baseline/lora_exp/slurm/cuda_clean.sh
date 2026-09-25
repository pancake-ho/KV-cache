#!/usr/bin/env bash

#SBATCH -J cuda-clean
#SBATCH -p batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=4G
#SBATCH --time=00:05:00
#SBATCH --export=NONE
#SBATCH -o runs/cuda-clean-%j.out
#SBATCH -e runs/cuda-clean-%j.err

set -u

echo "=== BASIC ==="
echo "host=$(hostname)"
echo "job=${SLURM_JOB_ID}"
echo "user=$(whoami)"
echo "uid=$(id -u)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

echo
echo "=== FULL ENV ==="
/usr/bin/env | sort

echo
echo "=== LIMITS ==="
ulimit -a

echo
echo "=== NVIDIA ==="
/usr/bin/nvidia-smi || nvidia-smi

echo
echo "=== DIRECT CUDA DRIVER TEST ==="

/usr/bin/python3 - <<'PY'
import ctypes
import ctypes.util

print("python clean system interpreter")

p = ctypes.util.find_library("cuda")
print("find_library(cuda):", p)

lib = ctypes.CDLL("libcuda.so.1")
print("libcuda load: PASS")

lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int

ret = lib.cuInit(0)

name = ctypes.c_char_p()
desc = ctypes.c_char_p()

lib.cuGetErrorName.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p)
]
lib.cuGetErrorString.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p)
]

lib.cuGetErrorName(ret, ctypes.byref(name))
lib.cuGetErrorString(ret, ctypes.byref(desc))

print("cuInit =", ret)
print("name   =", name.value.decode() if name.value else None)
print("desc   =", desc.value.decode() if desc.value else None)
PY