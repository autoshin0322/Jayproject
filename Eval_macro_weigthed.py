import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 이미지 저장 가능
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)


### Eval mit ConfusionMatrix, Macro, Weighted Avg

def evaluate_all_labeledeaf(
    base_dir="video1",
    save_confusion_csv=True,
    save_confusion_png=True,
    save_global_cm=True,           # 전체 합산 CM 저장 여부
    normalize_png=False            # PNG를 정규화해서 그릴지 여부
):
    label_dir = os.path.join(base_dir, "labeledeaf")
    pred_dir  = os.path.join(base_dir, "output")
    results   = []

    classes = ["Gesture", "NoGesture"]

    # 폴더 준비
    os.makedirs(base_dir, exist_ok=True)
    cm_dir = os.path.join(base_dir, "confusion_matrices")
    if save_confusion_csv or save_confusion_png:
        os.makedirs(cm_dir, exist_ok=True)

    # 글로벌(전체 합산) CM용 누적 변수
    global_cm = None

    for filename in sorted(os.listdir(label_dir)):
        if not filename.endswith(".csv"):
            continue

        # e.g., video1_camera.csv -> camera
        framename = filename.replace(f"{base_dir}_", "").replace(".csv", "")
        gt_path   = os.path.join(label_dir, filename)
        pred_path = os.path.join(pred_dir, f"{framename}.mp4_predictions.csv")

        if not os.path.exists(pred_path):
            print(f"[⚠] 예측 파일 없음: {framename}")
            continue

        # 필요한 컬럼만 사용
        df_gt   = pd.read_csv(gt_path)[["time", "label"]].rename(columns={"label":"label_true"})
        df_pred = pd.read_csv(pred_path)[["time", "label"]].rename(columns={"label":"label_pred"})

        # time 기준 inner join
        df = pd.merge(df_gt, df_pred, on="time", how="inner")
        if df.empty:
            print(f"[⚠] time 매칭된 샘플 없음: {framename}")
            continue

        y_true = df["label_true"].astype(str)
        y_pred = df["label_pred"].astype(str)

        # 파일별 Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # 글로벌 CM 누적
        if global_cm is None:
            global_cm = cm.copy()
        else:
            global_cm += cm

        # CSV 저장
        if save_confusion_csv:
            cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                                    columns=[f"pred_{c}" for c in classes])
            cm_path = os.path.join(cm_dir, f"{framename}_cm.csv")
            cm_df.to_csv(cm_path, index=True)

        # PNG 저장
        if save_confusion_png:
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            # 정규화 시각화 (행 기준 비율)
            if normalize_png:
                with pd.option_context('mode.use_inf_as_na', True):
                    row_sums = cm.sum(axis=1, keepdims=True)
                    cm_norm = cm / (row_sums + (row_sums == 0))  # 분모 0 보호
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
            disp.plot(cmap="Blues", ax=ax, colorbar=False)
            ax.set_title(f"Confusion Matrix: {framename}" + (" (normalized)" if normalize_png else ""))
            plt.tight_layout()
            out_png = os.path.join(cm_dir, f"{framename}_cm.png" if not normalize_png else f"{framename}_cm_norm.png")
            plt.savefig(out_png, dpi=200)
            plt.close(fig)

        # 클래스별 지표
        prec_g = precision_score(y_true, y_pred, labels=classes, pos_label="Gesture",   zero_division=0)
        rec_g  = recall_score(   y_true, y_pred, labels=classes, pos_label="Gesture",   zero_division=0)
        f1_g   = f1_score(       y_true, y_pred, labels=classes, pos_label="Gesture",   zero_division=0)

        prec_n = precision_score(y_true, y_pred, labels=classes, pos_label="NoGesture", zero_division=0)
        rec_n  = recall_score(   y_true, y_pred, labels=classes, pos_label="NoGesture", zero_division=0)
        f1_n   = f1_score(       y_true, y_pred, labels=classes, pos_label="NoGesture", zero_division=0)

        # 평균 지표
        prec_macro = precision_score(y_true, y_pred, average="macro",   zero_division=0)
        rec_macro  = recall_score(   y_true, y_pred, average="macro",   zero_division=0)
        f1_macro   = f1_score(       y_true, y_pred, average="macro",   zero_division=0)

        prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_w  = recall_score(   y_true, y_pred, average="weighted", zero_division=0)
        f1_w   = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

        acc = accuracy_score(y_true, y_pred)

        print(f"📊 {framename}")
        print(f"  [Gesture]     P:{prec_g:.3f}  R:{rec_g:.3f}  F1:{f1_g:.3f}")
        print(f"  [NoGesture]   P:{prec_n:.3f}  R:{rec_n:.3f}  F1:{f1_n:.3f}")
        print(f"  [Macro-Avg]   P:{prec_macro:.3f}  R:{rec_macro:.3f}  F1:{f1_macro:.3f}")
        print(f"  [Weighted-Avg]P:{prec_w:.3f}    R:{rec_w:.3f}    F1:{f1_w:.3f}")
        print(f"  Accuracy: {acc:.3f}\n")

        results.append({
            "framename": framename,
            # 클래스별
            "precision_Gesture":   prec_g,
            "recall_Gesture":      rec_g,
            "f1_Gesture":          f1_g,
            "precision_NoGesture": prec_n,
            "recall_NoGesture":    rec_n,
            "f1_NoGesture":        f1_n,
            # 평균
            "precision_macro":     prec_macro,
            "recall_macro":        rec_macro,
            "f1_macro":            f1_macro,
            "precision_weighted":  prec_w,
            "recall_weighted":     rec_w,
            "f1_weighted":         f1_w,
            # 정확도
            "accuracy":            acc
        })

    df_results = pd.DataFrame(results)

    # 전체 평균 행 추가
    if not df_results.empty:
        avg_row = {"framename": "AVERAGE"}
        for col in df_results.columns:
            if col != "framename":
                avg_row[col] = df_results[col].mean()
        df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)

    # ✅ 글로벌(전체 합산) CM 저장
    if save_global_cm and global_cm is not None:
        # CSV
        if save_confusion_csv:
            gcm_df = pd.DataFrame(global_cm, index=[f"true_{c}" for c in classes],
                                             columns=[f"pred_{c}" for c in classes])
            gcm_path = os.path.join(cm_dir, f"_GLOBAL_cm.csv")
            gcm_df.to_csv(gcm_path, index=True)

        # PNG
        if save_confusion_png:
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=classes)
            if normalize_png:
                row_sums = global_cm.sum(axis=1, keepdims=True)
                gcm_norm = global_cm / (row_sums + (row_sums == 0))
                disp = ConfusionMatrixDisplay(confusion_matrix=gcm_norm, display_labels=classes)
            disp.plot(cmap="Blues", ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix: GLOBAL" + (" (normalized)" if normalize_png else ""))
            plt.tight_layout()
            out_png = os.path.join(cm_dir, "_GLOBAL_cm.png" if not normalize_png else "_GLOBAL_cm_norm.png")
            plt.savefig(out_png, dpi=240)
            plt.close(fig)

    return df_results


# 실행 & 저장 예시
df_results = evaluate_all_labeledeaf(
    base_dir="video1",
    save_confusion_csv=True,
    save_confusion_png=True,   # PNG도 저장
    save_global_cm=True,       # 전체 합산 CM 저장
    normalize_png=False        # True면 비율로 시각화
)
df_results.to_csv("video1/evaluation_summary.csv", index=False)
print("✅ Saved: video1/evaluation_summary.csv")
print("✅ Confusion matrices in: video1/confusion_matrices/")