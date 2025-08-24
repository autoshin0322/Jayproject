import os
import pandas as pd

videoname = "video2"

def batch_generate_bio_labels(
    input_folder=f"{videoname}/labeledeaf",
    output_folder=f"{videoname}/bio_true",
    gap_threshold=0.5
):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue

        input_path = os.path.join(input_folder, filename)

        # 이름에서 framename 추출
        framename = filename.replace(videoname, "").replace(".csv", "")
        output_filename = f"{framename}.csv"
        output_path = os.path.join(output_folder, output_filename)

        # 라벨링
        df = pd.read_csv(input_path)
        df = df[["time", "label"]]

        times = df["time"].tolist()
        labels = df["label"].tolist()
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
            "label": labels,
            "BIO": bio_labels
        })

        df_out.to_csv(output_path, index=False)
        print(f"[✔] 저장 완료: {output_path}")


batch_generate_bio_labels(
    input_folder=f"{videoname}/labeledeaf", ### path to videos
    output_folder=f"{videoname}/bio_true"   ### path for outputs
)