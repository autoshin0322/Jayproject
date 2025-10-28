import os
import pandas as pd

features_dir = "data/features"
labels_dir = "data/labels"

pairs = []
for f in sorted(os.listdir(features_dir)):
    if f.endswith(".mp4_features.npy"):
        clip_id = f.replace(".mp4_features.npy", "")
        feature_path = os.path.join(features_dir, f)
        label_path = os.path.join(labels_dir, f"{clip_id}.csv")
        if os.path.exists(label_path):
            pairs.append({"feature_path": feature_path, "label_path": label_path})

df = pd.DataFrame(pairs)
df.to_csv("data/index_train.csv", index=False)
print(f"✅ Created data/index_train.csv with {len(df)} entries")
