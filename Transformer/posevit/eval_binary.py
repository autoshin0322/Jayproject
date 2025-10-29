# posevit/eval_binary.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def eval_pair(true_csv, pred_csv, out_png="outputs/figs/cm.png"):
    df_t = pd.read_csv(true_csv)
    df_p = pd.read_csv(pred_csv)

    y_true = df_t["label"].map({"NoGesture":0,"Gesture":1}).to_numpy()
    y_pred = df_p["label"].map({"NoGesture":0,"Gesture":1}).to_numpy()

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    P = precision_score(y_true, y_pred, zero_division=0)
    R = recall_score(y_true, y_pred, zero_division=0)
    F = f1_score(y_true, y_pred, zero_division=0)
    A = accuracy_score(y_true, y_pred)
    print(f"Precision={P:.4f} Recall={R:.4f} F1={F:.4f} Acc={A:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["NoGesture","Gesture"],
                yticklabels=["NoGesture","Gesture"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    print(f"CM saved: {out_png}")

if __name__ == "__main__":
    eval_pair(
        true_csv="data/TestLabels/000.csv",
        pred_csv="outputs/preds/000.mp4_predictions.csv",
        out_png="outputs/figs/000_cm.png"
    )
