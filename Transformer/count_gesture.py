import pandas as pd
import os

# CSV 파일들이 있는 폴더
base_dir = "data/labels"

# 파일 리스트
csv_files = [
    "M3D_TED_AS2010.csv",
    "M3D_TED_EG2009.csv",
    "M3D_TED_ES2016.csv",
    "M3D_TED_MS2010.csv",
    "M3D_TED_SJ2010.csv"
]

FPS = 32  # envisionHGDetector의 프레임레이트

# 전체 통계 저장용 변수
total_gesture_frames = 0
total_nogesture_frames = 0

print("=== 각 데이터셋별 Gesture / NoGesture 프레임 통계 (FPS 32 적용) ===\n")

for file_name in csv_files:
    file_path = os.path.join(base_dir, file_name)
    df = pd.read_csv(file_path)

    # duration(초) × 32 → 프레임 수로 변환
    df["frames"] = df["duration"] * FPS

    # 라벨별 프레임 합산
    gesture_frames = df[df["label"] == "Gesture"]["frames"].sum()
    nogesture_frames = df[df["label"] == "NoGesture"]["frames"].sum()
    total_frames = gesture_frames + nogesture_frames

    gesture_ratio = (gesture_frames / total_frames * 100) if total_frames > 0 else 0
    nogesture_ratio = (nogesture_frames / total_frames * 100) if total_frames > 0 else 0

    # 결과 출력
    print(f"📂 {file_name}")
    print(f"  Gesture: {gesture_frames:.0f} Frames ({gesture_ratio:.2f}%)")
    print(f"  NoGesture: {nogesture_frames:.0f} Frames ({nogesture_ratio:.2f}%)")
    print(f"  Total: {total_frames:.0f} Frames\n")

    total_gesture_frames += gesture_frames
    total_nogesture_frames += nogesture_frames

# 전체 합계
grand_total = total_gesture_frames + total_nogesture_frames
print("=== 전체 합계 (FPS 32 적용) ===")
print(f"Gesture: {total_gesture_frames:.0f} Frames ({(total_gesture_frames / grand_total) * 100:.2f}%)")
print(f"NoGesture: {total_nogesture_frames:.0f} Frames ({(total_nogesture_frames / grand_total) * 100:.2f}%)")
print(f"Total Frames: {grand_total:.0f}")
