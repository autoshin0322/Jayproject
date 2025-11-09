import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# === Binary label mapping ===
LABEL_MAP_BIN = {
    "Gesture": 1,
    "NoGesture": 0
}

# === 슬라이딩 윈도우 생성 함수 ===
def make_windows(X, y, T=32, S=16):
    """
    X: (frames, features)
    y: (frames,)
    T: window length
    S: stride
    return:
        Xs: (num_windows, T, features)
        ys: (num_windows, T)
        idxs: (num_windows,)
    """
    if len(X) < T:
        return np.empty((0, T, X.shape[1])), np.empty((0, T)), np.array([])

    Xs, ys, idxs = [], [], []
    for start in range(0, len(X) - T + 1, S):
        Xs.append(X[start:start + T])
        ys.append(y[start:start + T])
        idxs.append(start)
    return np.stack(Xs), np.stack(ys), np.array(idxs)


# === Dataset 클래스 정의 ===
class PoseSeqDataset(Dataset):
    def __init__(self, index_csv, T=32, S=16):
        """
        index_csv: feature_path, label_path 컬럼 포함
        T: window length
        S: stride
        """
        self.data = []
        df = pd.read_csv(index_csv)

        for _, r in df.iterrows():
            f_path, l_path = r["feature_path"], r["label_path"]

            # feature, label 로드
            if not os.path.exists(f_path) or not os.path.exists(l_path):
                print(f"[⚠️ missing file] {f_path} or {l_path}")
                continue

            X = np.load(f_path)
            df_label = pd.read_csv(l_path)

            # label 컬럼명 자동 탐지
            label_col = None
            for c in df_label.columns:
                if c.lower() in ["label", "bio", "gesture"]:
                    label_col = c
                    break
            if label_col is None:
                print(f"[⚠️ no valid label column] {l_path}")
                continue

            y = df_label[label_col].map(LABEL_MAP_BIN).fillna(0).astype(int).values

            # 길이 일치 보정
            min_len = min(len(X), len(y))
            X, y = X[:min_len], y[:min_len]

            # 길이별 처리
            if len(X) < 27:
                continue
            elif len(X) < T:
                pad_len = T - len(X)
                X = np.pad(X, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
                y = np.pad(y, (0, pad_len), mode="constant", constant_values=0)

            # 윈도우 생성
            Xw, yw, starts = make_windows(X, y, T, S)
            if len(Xw) == 0:
                continue

            for Xi, yi in zip(Xw, yw):
                self.data.append((Xi, yi))

        print(f"✅ Loaded {len(self.data)} samples from {len(df)} clips.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
