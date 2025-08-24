import os
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate_all_bio(
    true_folder="video2/bio_true",
    pred_folder="video2/bio_pred",
    output_folder="video2/bio_eval"
):
    os.makedirs(output_folder, exist_ok=True)
    results = []

    all_y_true = []
    all_y_pred = []

    for filename in os.listdir(true_folder):
        if not filename.endswith(".csv"):
            continue

        # bio_000.csv → 000
        framename = filename.replace("bio_", "").replace(".csv", "")
        true_path = os.path.join(true_folder, filename)

        # pred 쪽 파일명: video2_000.csv
        pred_filename = f"video2_{framename}.csv"
        pred_path = os.path.join(pred_folder, pred_filename)

        if not os.path.exists(pred_path):
            print(f"[⚠] 예측 파일 없음: {pred_filename}")
            continue

        df_true = pd.read_csv(true_path)[["time", "BIO"]].rename(columns={"BIO": "BIO_true"})
        df_pred = pd.read_csv(pred_path)[["time", "BIO"]].rename(columns={"BIO": "BIO_pred"})
        df = pd.merge(df_true, df_pred, on="time")

        y_true = df["BIO_true"]
        y_pred = df["BIO_pred"]

        # 누적 저장
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # 지표 계산
        precision = precision_score(y_true, y_pred, average="macro", labels=["B", "I", "O"], zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", labels=["B", "I", "O"], zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", labels=["B", "I", "O"], zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        results.append({
            "framename": framename,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        })

        # 개별 confusion matrix 저장
        cm = confusion_matrix(y_true, y_pred, labels=["B", "I", "O"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["B", "I", "O"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix: {framename}")
        plt.tight_layout()
        cm_path = os.path.join(output_folder, f"cm_{framename}.png")
        plt.savefig(cm_path)
        plt.close()

    # 전체 confusion matrix 저장
    if all_y_true and all_y_pred:
        cm_all = confusion_matrix(all_y_true, all_y_pred, labels=["B", "I", "O"])
        disp_all = ConfusionMatrixDisplay(cm_all, display_labels=["B", "I", "O"])
        disp_all.plot(cmap="Purples", values_format="d")
        plt.title("Confusion Matrix: ALL")
        plt.tight_layout()
        cm_all_path = os.path.join(output_folder, "cm_ALL.png")
        plt.savefig(cm_all_path)
        plt.close()

    # 정렬 + 평균
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="framename")

    if not df_results.empty:
        avg_row = {
            "framename": "AVERAGE",
            "precision": df_results["precision"].mean(),
            "recall": df_results["recall"].mean(),
            "f1": df_results["f1"].mean(),
            "accuracy": df_results["accuracy"].mean()
        }
        df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)

    # 저장
    summary_csv = os.path.join(output_folder, "summary.csv")
    df_results.to_csv(summary_csv, index=False)

    return summary_csv, output_folder

# 실행
evaluate_all_bio(
    true_folder="video2/bio_true",
    pred_folder="video2/bio_pred",
    output_folder="video2/bio_eval"
)