import os
import numpy as np
from collections import defaultdict

features_dir = "data/features"

# Dictionaries to store results
total_frames_by_file = defaultdict(int)
short_frames_by_file = defaultdict(int)

# Loop through feature files
for f in sorted(os.listdir(features_dir)):
    if not f.endswith(".npy"):
        continue

    # Example filename: 3Educator_id2_politics_vid1_000.mp4_features.npy
    filename = "_".join(f.split("_")[:-2])  # remove "_index.mp4_features.npy"
    path = os.path.join(features_dir, f)

    arr = np.load(path)
    n_frames = arr.shape[0]

    total_frames_by_file[filename] += n_frames
    if n_frames < 27:
        short_frames_by_file[filename] += n_frames

# Print results
print("📊 Frame Summary by File:\n")
summary = []
grand_total_valid = 0

for filename in sorted(total_frames_by_file.keys()):
    a = total_frames_by_file[filename]
    b = short_frames_by_file.get(filename, 0)
    c = a - b
    grand_total_valid += c
    summary.append((filename, a, b, c))
    print(f"{filename:<40} total_frames(a)={a:>6} | short_frames(<27)(b)={b:>6} | valid_frames(c)={c:>6}")

print(f"\n📈 Total valid frames across all files: {grand_total_valid}")
