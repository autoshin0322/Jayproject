import torch
import numpy as np
from posevit.model import PoseSeqTransformer

# --- 설정 ---
model_path = "outputs/ckpts/posevit_best.pt"
feature_path = "data/features_test/input.mp4_features.npy"
output_path = "outputs/predictions.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 데이터 로드 ---
X = np.load(feature_path)
print(f"✅ Loaded features {X.shape}")

# --- 윈도우 분할 파라미터 ---
T = 32   # sequence length
S = 32   # stride

def make_windows(X, T, S):
    out = []
    for i in range(0, len(X) - T + 1, S):
        out.append(X[i:i+T])
    return np.stack(out)

windows = make_windows(X, T, S)
print(f"✅ Created {len(windows)} windows")

# --- 모델 로드 ---
model = PoseSeqTransformer(d_in=80, d_model=128, nhead=8, num_layers=2, num_classes=2)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)

# --- 예측 ---
preds = []
with torch.no_grad():
    for w in windows:
        x = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 80)
        logits = model(x)                    # (1, T, 2)
        last_logits = logits[:, -1, :]       # 마지막 timestep의 [NoGesture, Gesture]
        probs = torch.softmax(last_logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        preds.append(pred)

preds = np.array(preds)
np.savetxt(output_path, preds, fmt="%d")
print(f"✅ Saved predictions → {output_path}")
print(f"🔍 Distribution: 0={np.sum(preds==0)}, 1={np.sum(preds==1)}")
