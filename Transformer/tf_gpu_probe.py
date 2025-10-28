# save as: tf_gpu_probe.py  (python tf_gpu_probe.py)
import os, sys, ctypes, glob
import tensorflow as tf

print("TF:", tf.__version__)
try:
    bi = tf.sysconfig.get_build_info()
    print("Build CUDA:", bi.get("cuda_version"))
    print("Build cuDNN:", bi.get("cudnn_version"))
except Exception as e:
    print("build_info not available:", e)

print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

# 어떤 라이브러리가 안 열리는지 직접 확인
candidates = [
    "libcudart.so", "libcudart.so.11.0",
    "libcublas.so", "libcublas.so.11",
    "libcudnn.so", "libcudnn.so.8",
    "libcusolver.so", "libcusolver.so.11",
    "libcufft.so", "libcufft.so.10",
]
print("\n[DLopen test]")
ok = True
for name in candidates:
    try:
        ctypes.CDLL(name)
        print("  OK  ", name)
    except OSError as e:
        print("  FAIL", name, "->", e)
        ok = False

print("\nCONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
if os.environ.get("CONDA_PREFIX"):
    libdir = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    print("Listing CUDA-ish libs in", libdir)
    for p in sorted(glob.glob(os.path.join(libdir, "libcu*d*.so*"))):
        print("  ", os.path.basename(p))

if not ok:
    print("\n=> 위 FAIL 항목이 GPU 미인식의 직접 원인이야.")
