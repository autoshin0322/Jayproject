import numpy as np, os

features_dir = "data/features"
for f in sorted(os.listdir(features_dir)):
    if f.endswith(".npy"):
        arr = np.load(os.path.join(features_dir, f))
        print(f, arr.shape)
