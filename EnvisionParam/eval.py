import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# === 기본 설정 ===
path = "test"  # 폴더 경로
true_df = pd.read_csv(f"{path}/annotation.csv")
pred_df = pd.read_csv(f"{path}/output/input.mp4_segments.csv")

# === FPS 설정 ===
FPS = 32
timestep = 1 / FPS

# === 사용자 지정 시간 범위 (초 단위) ===
START_TIME = 608 # 분석 시작 시점 (예: 600초)
END_TIME = 1472    # 분석 종료 시점 (예: 1200초)

# === ⏱️ 시간 보정: input.mp4_segments.csv → 전역 타임라인으로 shift ===
pred_df["start_time"] = pred_df["start_time"] + START_TIME
pred_df["end_time"] = pred_df["end_time"] + START_TIME

# === 전체 시간 구간 생성 (지정된 구간만 평가) ===
timeline = np.arange(START_TIME, END_TIME, timestep)

# === Ground Truth / Prediction 초기화 ===
y_true = np.zeros(len(timeline))
y_pred = np.zeros(len(timeline))

# === Ground Truth: 사람이 지정한 구간 ===
for _, row in true_df.iterrows():
    start, end = row["start_time"], row["end_time"]
    # 지정된 시간 범위와 겹치는 구간만 반영
    if end < START_TIME or start > END_TIME:
        continue
    mask = (timeline >= start) & (timeline <= end)
    y_true[mask] = 1

# === Model Prediction: 모델이 감지한 구간 ===
for _, row in pred_df.iterrows():
    start, end = row["start_time"], row["end_time"]
    if end < START_TIME or start > END_TIME:
        continue
    mask = (timeline >= start) & (timeline <= end)
    y_pred[mask] = 1

# === 성능 평가 ===
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)

print(f"⏱ 평가 구간: {START_TIME:.1f}s ~ {END_TIME:.1f}s")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

# === Confusion Matrix 계산 ===
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["True_NoGesture", "True_Gesture"],
    columns=["Pred_NoGesture", "Pred_Gesture"]
)

# === Confusion Matrix 시각화 ===
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues")
plt.title(f"Confusion Matrix ({START_TIME}-{END_TIME}s)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{path}/output/confusion_matrix.png", dpi=300)
plt.close()
print(f"🖼️ Confusion Matrix gespeichert: {path}/output/confusion_matrix.png")

# === 프레임 단위 비교 CSV 생성 ===
comparison_df = pd.DataFrame({
    "time(sec)": timeline,
    "Ground Truth": y_true.astype(int),
    "Prediction": y_pred.astype(int)
})
comparison_path = f"{path}/output/groundtruth_vs_prediction.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"📄 Vergleich CSV gespeichert: {comparison_path}")

# === 평가 결과 저장 ===
pd.DataFrame({
    "metric": ["precision", "recall", "f1", "accuracy"],
    "value": [precision, recall, f1, accuracy]
}).to_csv(f"{path}/output/evaluation.csv", index=False)
print(f"✅ Ergebnisse gespeichert in {path}/output/evaluation.csv")