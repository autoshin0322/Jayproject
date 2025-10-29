import os
import numpy as np
import pandas as pd
import torch
from model import PoseSeqTransformer


@torch.no_grad()
def predict_file(
    ckpt_path,
    feature_path,
    fps=25,
    T=32,
    S=16,
    out_csv="outputs/preds/pred.csv",
    device=None,
    thr=0.5
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")

    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    cfg = ckpt["cfg"]

    model = PoseSeqTransformer(
        d_in=cfg["d_in"],
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 4),
        num_classes=cfg.get("num_classes", 2),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    X = np.load(feature_path).astype(np.float32)
    N, F = X.shape

    starts = list(range(0, N - T + 1, S))
    prob_acc = np.zeros((N, 2), dtype=np.float32)
    count = np.zeros((N, 1), dtype=np.float32)

    for st in starts:
        chunk = torch.from_numpy(X[st:st + T]).unsqueeze(0).to(device)
        logits = model(chunk)
        p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        prob_acc[st:st + T] += p
        count[st:st + T] += 1

    mask = (count > 0).reshape(-1)
    prob = np.zeros_like(prob_acc)
    prob[mask] = prob_acc[mask] / count[mask]
    pred = (prob[:, 1] >= thr).astype(int)

    times = np.arange(N) / fps
    labels = np.where(pred == 1, "Gesture", "NoGesture")
    out = pd.DataFrame({"time": times, "label": labels, "Gesture_prob": prob[:, 1]})
    out.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")


if __name__ == "__main__":
    ckpt_path = "outputs/ckpts/posevit_best.pt"
    feature_dir = "data/TestFeatures"
    output_dir = "outputs/preds"
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 폴더 내 모든 .mp4_features.npy 파일 처리
    files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".mp4_features.npy"))

    print(f"[INFO] Found {len(files)} feature files in {feature_dir}")

    for f in files:
        feature_path = os.path.join(feature_dir, f)
        out_csv = os.path.join(output_dir, f.replace(".npy", "_pred.csv"))
        print(f"[RUN] Predicting {f} ...")
        predict_file(
            ckpt_path=ckpt_path,
            feature_path=feature_path,
            fps=25,
            T=32,
            S=16,
            out_csv=out_csv,
            device=None,
            thr=0.5
        )

    print("✅ All predictions completed!")
