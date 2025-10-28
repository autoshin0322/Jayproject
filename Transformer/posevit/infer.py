# posevit/infer.py
import os, numpy as np, pandas as pd, torch
from model import PoseSeqTransformer

@torch.no_grad()
def predict_file(ckpt_path, feature_path, fps=32, T=32, S=16, out_csv="outputs/preds/pred.csv", device="cuda:0", thr=0.5):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    d_in = ckpt["cfg"]["d_in"]
    model = PoseSeqTransformer(d_in=d_in).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    X = np.load(feature_path).astype(np.float32)  # (N,F)
    N, F = X.shape
    # windowing
    starts = list(range(0, N-T+1, S))
    prob_acc = np.zeros((N,2), dtype=np.float32)
    count    = np.zeros((N,1), dtype=np.float32)

    for st in starts:
        chunk = torch.from_numpy(X[st:st+T]).unsqueeze(0).to(device)  # (1,T,F)
        logits = model(chunk)                                         # (1,T,2)
        p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()    # (T,2)
        prob_acc[st:st+T] += p
        count[st:st+T]    += 1

    # 평균
    mask = (count>0).reshape(-1)
    prob = np.zeros_like(prob_acc)
    prob[mask] = prob_acc[mask] / count[mask]
    # 빈 구간은 가장 가까운 예측을 복제하거나 NoGesture로 둠(여기선 0)
    pred = (prob[:,1] >= thr).astype(int)

    # CSV (Envision 포맷)
    times = np.arange(N)/fps
    labels = np.where(pred==1, "Gesture", "NoGesture")
    out = pd.DataFrame({"time": times, "label": labels, "Gesture_prob": prob[:,1]})
    out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")

if __name__ == "__main__":
    predict_file(
        ckpt_path="outputs/ckpts/posevit_best.pt",
        feature_path="data/features/000_features.npy",
        fps=32, T=32, S=16,
        out_csv="outputs/preds/000.mp4_predictions.csv",
        device="cuda:0", thr=0.5
    )
