# posevit/train.py
import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import PoseSeqTransformer
from dataset import PoseSeqDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train(index_csv, d_in, out_dir="outputs", T=32, S=16, epochs=20, bs=16, lr=3e-4, device="cuda:0"):
    os.makedirs(out_dir+"/ckpts", exist_ok=True)
    ds = PoseSeqDataset(index_csv, T=T, S=S)

    # 90/10 split
    n_val = max(1, int(len(ds)*0.1))
    n_train = len(ds)-n_val
    ds_tr, ds_va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = PoseSeqTransformer(d_in=d_in, d_model=128, nhead=4, num_layers=4, num_classes=2).to(device)
    # 클래스 불균형 가중치(필요 시 비율에 맞춰 조정)
    weight = torch.tensor([1.0, 3.0], device=device)  # [NoGesture, Gesture]
    crit = nn.CrossEntropyLoss(weight=weight)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0.0
    for ep in range(1, epochs+1):
        # --- train
        model.train(); tr_loss=0
        for Xb, yb in tqdm(dl_tr, desc=f"Train {ep}"):
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)                 # (B,T,2)
            loss = crit(logits.reshape(-1,2), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()*Xb.size(0)

        # --- val
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for Xb, yb in dl_va:
                Xb = Xb.to(device)
                logits = model(Xb)             # (B,T,2)
                pred = logits.argmax(-1).cpu().numpy().reshape(-1)
                ys.extend(yb.numpy().reshape(-1).tolist())
                ps.extend(pred.tolist())

        f1  = f1_score(ys, ps, average="binary", zero_division=0)
        pre = precision_score(ys, ps, zero_division=0)
        rec = recall_score(ys, ps, zero_division=0)
        acc = accuracy_score(ys, ps)
        print(f"[Val] ep{ep} F1={f1:.4f} P={pre:.4f} R={rec:.4f} Acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            ckpt = os.path.join(out_dir, "ckpts", f"posevit_best.pt")
            torch.save({"model":model.state_dict(),
                        "cfg":{"d_in":d_in,"T":T,"S":S}}, ckpt)
            print(f"✓ saved {ckpt}")

if __name__ == "__main__":
    # d_in: feature 차원(예: 63)
    train(index_csv="data/index_train.csv", d_in=80, out_dir="outputs", T=32, S=16, epochs=20, bs=16, lr=3e-4, device="cuda:0")
