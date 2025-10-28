import numpy as np
features = np.load("input.mp4_features.npy")
fps = 32
print(f"총 프레임: {len(features)}")
print(f"추정 영상 길이: {len(features) / fps:.2f}초")
