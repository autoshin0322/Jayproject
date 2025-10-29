# posevit/eval_binary.py (TestLabels 폴더 호환 버전)
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)

# =====================================
# 단일 파일 평가
# =====================================
def eval_pair(true_csv, pred_csv, out_png=None):
    df_t = pd.read_csv(true_csv)
    df_p = pd.read_csv(pred_csv)

    # 라벨을 숫자(0/1)로 변환
    y_true = df_t["label"].map({"NoGesture": 0, "Gesture": 1}).to_numpy()
    y_pred = df_p["label"].map({"NoGesture": 0, "Gesture": 1}).to_numpy()

    # 두 시퀀스 길이 맞추기
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    # 메트릭 계산
    P = precision_score(y_true, y_pred, zero_division=0)
    R = recall_score(y_true, y_pred, zero_division=0)
    F = f1_score(y_true, y_pred, zero_division=0)
    A = accuracy_score(y_true, y_pred)

    # Confusion Matrix 저장
    if out_png:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["NoGesture", "Gesture"],
            yticklabels=["NoGesture", "Gesture"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(os.path.basename(pred_csv))
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

    return {"precision": P, "recall": R, "f1": F, "accuracy": A}


# =====================================
# 전체 예측 파일 평가
# =====================================
def eval_all(pred_dir="outputs/preds", label_dir="data/TestLabels", report_dir="outputs/reports"):
    os.makedirs(report_dir, exist_ok=True)
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.mp4_features_predictions.csv")))

    if not pred_files:
        print(f"[⚠️ No prediction files found in {pred_dir}]")
        return

    results = []
    print(f"🔍 Found {len(pred_files)} prediction files in {pred_dir}\n")

    for pred_csv in pred_files:
        base = os.path.basename(pred_csv).replace(".mp4_features_predictions.csv", "")
        true_csv = os.path.join(label_dir, f"{base}.csv")
        out_png = os.path.join(report_dir, f"{base}_cm.png")

        if not os.path.exists(true_csv):
            print(f"[⚠️ Missing ground truth] {true_csv}")
            continue

        try:
            metrics = eval_pair(true_csv, pred_csv, out_png)
            metrics["video"] = base
            results.append(metrics)
            print(f"✅ Evaluated {base}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"[❌ Failed] {base}: {e}")

    if not results:
        print("[⚠️ No valid evaluation results found.]")
        return

    # 결과 DataFrame 생성 및 평균 계산
    df = pd.DataFrame(results)
    mean_row = {col: df[col].mean() for col in ["precision", "recall", "f1", "accuracy"]}
    mean_row["video"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    report_path = os.path.join(report_dir, "eval_summary.csv")
    df.to_csv(report_path, index=False)

    print(f"\n📊 Summary saved to: {report_path}")
    print(df.tail())

    return df


if __name__ == "__main__":
    eval_all()
