import numpy as np
import pandas as pd

# --- 설정 ---
pred_path = "outputs/predictions.csv"   # 예측 결과
output_csv = "outputs/pred.csv" # 저장할 파일
fps = 32                                         # 영상 FPS
T = 32                                            # 윈도우 길이
S = 32                                            # stride

# --- 예측 로드 ---
preds = np.loadtxt(pred_path, delimiter=",", dtype=int)

# --- 정수 → 라벨 변환 ---
label_map = {0: "NoGesture", 1: "Gesture"}

intervals = []
for i, label in enumerate(preds):
    start_frame = i * S
    end_frame = start_frame + T
    start_time = start_frame / fps
    end_time = end_frame / fps
    label_name = label_map.get(label, "Unknown")
    intervals.append((i, label_name, start_time, end_time))

df = pd.DataFrame(intervals, columns=["index", "label", "start_time", "end_time"])

# --- 같은 label이 연속된 구간 병합 ---
merged = []
current_label = df.iloc[0]["label"]
start_time = df.iloc[0]["start_time"]
end_time = df.iloc[0]["end_time"]

for _, row in df.iloc[1:].iterrows():
    if row["label"] == current_label:
        end_time = row["end_time"]
    else:
        merged.append((current_label, start_time, end_time))
        current_label = row["label"]
        start_time = row["start_time"]
        end_time = row["end_time"]

# 마지막 구간 추가
merged.append((current_label, start_time, end_time))

# --- 결과 저장 ---
merged_df = pd.DataFrame(merged, columns=["label", "start_time", "end_time"])
merged_df.to_csv(output_csv, index=False)

print(f"✅ Saved labeled gesture intervals → {output_csv}")
print(merged_df.head(10))
