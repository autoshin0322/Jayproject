# automation/run_pipeline.py
import os
import subprocess
from pathlib import Path
import pandas as pd
import torch
from posevit.model import PoseViT
from posevit.dataset import PoseSeqDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===== 설정 =====
BASE = Path(__file__).resolve().parents[1]
VIDEOS = BASE / "videos_to_label"
ENVISION_OUT = BASE / "output_envision"
INDEX_ALL = BASE / "data" / "index_all.csv"
CKPT = BASE / "posevit" / "outputs" / "ckpts" / "posevit_best.pt"
RESULTS = BASE / "automation" / "results"
RESULTS.mkdir(exist_ok=True)

# ===== 1️⃣ Envision 실행 =====
print("▶ Running Envision feature extraction...")
subprocess.run([
    "python",
    str(BASE / "envision_extract.py"),  # 이 파일은 Envision에서 features만 추출하도록 만든 버전
    "--input", str(VIDEOS),
    "--output", str(ENVISION_OUT)
])

# ===== 2️⃣ index_all 생성 =====
print("▶ Building index_all.csv ...")
from make_index_all import build_index_all
build_index_all(ENVISION_OUT, INDEX_ALL)

# ===== 3️⃣ ViT 추론 =====
print("▶ Running ViT inference ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PoseViT(d_in=80, d_model=128, nhead=8, num_layers=4, num_classes=2)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device)
model.eval()

# index_all.csv 읽기
df = pd.read_csv(INDEX_ALL)
all_preds, all_trues = [], []

for _, row in df.iterrows():
    X = np.load(row["feature_path"])
    y = np.load(row["label_path"])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(X)
        pred = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
    all_preds.extend(pred)
    all_trues.extend(y)

# ===== 4️⃣ 평가 & 시각화 =====
print("▶ Evaluating ...")
report = classification_report(all_trues, all_preds, output_dict=True)
pd.DataFrame(report).to_csv(RESULTS / "metrics.csv")

cm = confusion_matrix(all_trues, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(RESULTS / "confusion_matrix.png")

print("✅ Pipeline completed. Results in:", RESULTS)
