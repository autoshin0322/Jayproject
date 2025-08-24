import os
import pandas as pd

videoname = "video2"

def batch_generate_bio_from_predictions(
    input_folder=f"{videoname}/output",
    output_folder=f"{videoname}/bio_pred",
    gap_threshold=0.5
):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".mp4_predictions.csv"):
            continue

        input_path = os.path.join(input_folder, filename)
        framename = filename.replace(".mp4_predictions.csv", "")
        output_filename = f"{framename}.csv"
        output_path = os.path.join(output_folder, output_filename)

        # 파일 로드
        df = pd.read_csv(input_path)
        if "time" not in df.columns or "label" not in df.columns:
            print(f"[⚠] 건너뜀 (time/label 없음): {filename}")
            continue

        df = df[["time", "label"]]

        # === 라벨 표준화: move -> Gesture, 기타는 Gesture/NoGesture로 맵 ===
        labels = (
            df["label"].astype(str).str.strip().str.lower()
              .replace({"move": "gesture"})                    # move를 gesture로 흡수
              .map(lambda x: "Gesture" if x == "gesture" else "NoGesture")
              .tolist()
        )
        times = df["time"].tolist()
        n = len(labels)

        bio_labels = []
        i = 0

        while i < n:
            current_label = labels[i]

            if current_label == "Gesture":
                if i == 0 or bio_labels[-1] == "O":
                    bio_labels.append("B")
                else:
                    bio_labels.append("I")
                i += 1

            elif current_label == "NoGesture":
                j = i + 1
                while j < n and labels[j] == "NoGesture":
                    j += 1

                if j < n and labels[j] == "Gesture":
                    time_gap = times[j] - times[i]
                    if bio_labels and bio_labels[-1] in ["B", "I"] and time_gap < gap_threshold:
                        bio_labels.append("I")
                        i += 1
                        continue

                bio_labels.append("O")
                i += 1

        # 저장
        df_out = pd.DataFrame({
            "time": times,
            "label": labels,   # 표준화된 라벨 저장
            "BIO": bio_labels
        })

        df_out.to_csv(output_path, index=False)
        print(f"[✔] 저장 완료: {output_path}")

batch_generate_bio_from_predictions(
    input_folder=f"{videoname}/output",      ### path to inputs .csv
    output_folder=f"{videoname}/bio_pred"    ### path for outputs
)