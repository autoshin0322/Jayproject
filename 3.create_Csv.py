import os
import pandas as pd
from pympi.Elan import Eaf

# path
videoname = "video2"                 # whole file path
eaf_dir = f"{videoname}/annotations" # Input .eaf
csv_dir = f"{videoname}/output"      # standard .csv
output_dir = f"{videoname}/labeledeaf" # output path

os.makedirs(output_dir, exist_ok=True)

# 모든 .eaf 파일에 대해 반복
for eaf_filename in os.listdir(eaf_dir):
    if not eaf_filename.endswith(".eaf"):
        continue

    framename = os.path.splitext(eaf_filename)[0]  # "camera" 등

    eaf_path = os.path.join(eaf_dir, eaf_filename)
    csv_reference_path = os.path.join(csv_dir, f"video2_{framename}.mp4_predictions.csv")
    output_csv_path = os.path.join(output_dir, f"{framename}.csv")

    # CSV 존재 확인
    if not os.path.exists(csv_reference_path):
        print(f"[⚠] CSV 없음: {csv_reference_path}")
        continue

    # time 추출
    csv_reference = pd.read_csv(csv_reference_path)
    time_points = csv_reference["time"].tolist()

    # EAF 로드 및 annotation 구간 수집
    eaf = Eaf(eaf_path)
    annotation_intervals = []
    for tier in eaf.get_tier_names():
        for start, end, value in eaf.get_annotation_data_for_tier(tier):
            annotation_intervals.append((start / 1000, end / 1000))  # ms → s

    # 라벨 결정 함수
    def get_label(t):
        for start, end in annotation_intervals:
            if start <= t <= end:
                return "Gesture"
        return "NoGesture"

    # 변환된 CSV 생성
    converted_df = pd.DataFrame({
        "time": time_points,
        "has_motion": 0,
        "NoGesture_confidence": 0,
        "Gesture_confidence": 0,
        "Move_confidence": 0,
        "label": [get_label(t) for t in time_points]
    })

    converted_df.to_csv(output_csv_path, index=False)
    print(f"✅ 변환 완료: {output_csv_path}")