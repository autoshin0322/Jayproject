# posevit/infer.py (배치 추론 버전)
import os, glob, numpy as np, pandas as pd, torch
from model import PoseSeqTransformer

@torch.no_grad()
def predict_file(ckpt_path, feature_path, fps=32, T=32, S=16, out_csv="outputs/preds/pred.csv", device="cuda:0", thr=0.5):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location=device)

    # === cfg에서 설정 읽기 ===
    cfg = ckpt.get("cfg", {})
    d_in = cfg.get("d_in", 80)
    d_model = cfg.get("d_model", 128)
    nhead = cfg.get("nhead", 4)
    num_layers = cfg.get("num_layers", 2)
    num_classes = cfg.get("num_classes", 2)

    # === 모델 초기화 ===
    model = PoseSeqTransformer(
        d_in=d_in,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # === Feature 불러오기 ===
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
    print(f"✅ Saved prediction: {out_csv}")


# =====================================
# 배치 추론: data/features_test/*.npy 전체 처리
# =====================================
def batch_infer(
    ckpt_path="outputs/ckpts/posevit_best.pt",
    feature_dir="data/TestFeatures",
    pred_dir="outputs/preds",
    fps=32,
    T=32,
    S=16,
    device="cuda:0",
    thr=0.5
):
    os.makedirs(pred_dir, exist_ok=True)
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))

    if not feature_files:
        print(f"[⚠️ No .npy files found in {feature_dir}]")
        return

    print(f"🔍 Found {len(feature_files)} feature files in {feature_dir}")

    for fpath in feature_files:
        base = os.path.basename(fpath).replace(".npy", "")
        out_csv = os.path.join(pred_dir, f"{base}_predictions.csv")
        try:
            predict_file(
                ckpt_path=ckpt_path,
                feature_path=fpath,
                fps=fps,
                T=T,
                S=S,
                out_csv=out_csv,
                device=device,
                thr=thr
            )
        except Exception as e:
            print(f"[❌ Failed] {base}: {e}")

    print(f"\n🎬 All predictions saved to: {pred_dir}")


if __name__ == "__main__":
    batch_infer()
