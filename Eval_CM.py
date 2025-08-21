import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 이미지 저장
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)

def evaluate_with_confusion(base_dir="D_21"):
    label_dir = os.path.join(base_dir, "labeledeaf")
    pred_dir = os.path.join(base_dir, "output")
    results = []
    cm_dir = os.path.join(base_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    classes = ["Gesture", "NoGesture"]
    global_cm = None  # 전체 합산 CM 누적용

    for filename in sorted(os.listdir(label_dir)):
        if not filename.endswith(".csv"):
            continue

        framename = filename.replace(f"{base_dir}_", "").replace(".csv", "")
        gt_path   = os.path.join(label_dir, filename)
        pred_path = os.path.join(pred_dir, f"{framename}.mp4_predictions.csv")

        if not os.path.exists(pred_path):
            print(f"[⚠] 예측 파일 없음: {framename}")
            continue

        df_gt   = pd.read_csv(gt_path)[["time", "label"]].rename(columns={"label":"label_true"})
        df_pred = pd.read_csv(pred_path)[["time", "label"]].rename(columns={"label":"label_pred"})
        df = pd.merge(df_gt, df_pred, on="time", how="inner")

        if df.empty:
            print(f"[⚠] 라벨과 예측이 매칭되지 않음: {framename}")
            continue

        y_true = df["label_true"].astype(str)
        y_pred = df["label_pred"].astype(str)

        precision = precision_score(y_true, y_pred, pos_label="Gesture", zero_division=0)
        recall    = recall_score(y_true, y_pred, pos_label="Gesture", zero_division=0)
        f1        = f1_score(y_true, y_pred, pos_label="Gesture", zero_division=0)
        accuracy  = accuracy_score(y_true, y_pred)

        print(f"📊 {framename}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-score:  {f1:.3f}")
        print(f"  Accuracy:  {accuracy:.3f}\n")

        # ✅ per-file CM 계산/저장 (PNG)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix: {framename}")
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f"{framename}_cm.png"), dpi=200)
        plt.close()

        # (선택) per-file CM CSV도 저장하고 싶으면 아래 2줄 주석 해제
        # pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
        #              columns=[f"pred_{c}" for c in classes]).to_csv(os.path.join(cm_dir, f"{framename}_cm.csv"))

        # ✅ 글로벌 CM 누적
        global_cm = cm if global_cm is None else (global_cm + cm)

        results.append({
            "framename": framename,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        })

    df_results = pd.DataFrame(results)

    # ✅ 전체 평균 행 추가
    if not df_results.empty:
        avg_row = {"framename": "AVERAGE"}
        for col in df_results.columns:
            if col != "framename":
                avg_row[col] = df_results[col].mean()
        df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)

        print("📈 전체 평균 결과")
        print(f"  Precision: {avg_row['precision']:.3f}")
        print(f"  Recall:    {avg_row['recall']:.3f}")
        print(f"  F1-score:  {avg_row['f1']:.3f}")
        print(f"  Accuracy:  {avg_row['accuracy']:.3f}\n")

    # ✅ 글로벌(전체 합산) CM 저장 (CSV + PNG 원본/정규화)
    if global_cm is not None:
        # CSV
        gcm_df = pd.DataFrame(global_cm, index=[f"true_{c}" for c in classes],
                                           columns=[f"pred_{c}" for c in classes])
        gcm_df.to_csv(os.path.join(cm_dir, "_GLOBAL_cm.csv"), index=True)

        # PNG (counts)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=classes)
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix: GLOBAL (counts)")
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, "_GLOBAL_cm.png"), dpi=240)
        plt.close(fig)

        # PNG (row-normalized, 비율)
        row_sums = global_cm.sum(axis=1, keepdims=True)
        gcm_norm = global_cm / (row_sums + (row_sums == 0))
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=gcm_norm, display_labels=classes)
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix: GLOBAL (row-normalized)")
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, "_GLOBAL_cm_norm.png"), dpi=240)
        plt.close(fig)

    return df_results


if __name__ == "__main__":
    df_results = evaluate_with_confusion("D_21")
    df_results.to_csv("D_21/evaluation_summary.csv", index=False)
    print("✅ CSV + Per-file CM PNG + Global CM(CSV/PNG) 저장 완료")