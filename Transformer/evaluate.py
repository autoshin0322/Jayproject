# evaluate.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from posevit.model import PoseSeqTransformer as PoseViT
from posevit.dataset import PoseSeqDataset

def evaluate(model_path, index_csv, d_in=80, T=32, S=16, device="cuda:0"):
    ds = PoseSeqDataset(index_csv, T=T, S=S)
    dl = DataLoader(ds, batch_size=16, shuffle=False)
    
    model = PoseViT(d_in=d_in, d_model=128, nhead=8, num_layers=2, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_true, all_pred = [], []

    with torch.no_grad():
        for X, y in dl:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_pred.extend(preds.flatten())
            all_true.extend(y.cpu().numpy().flatten())

    # === Classification Report ===
    report = classification_report(all_true, all_pred, target_names=["NoGesture", "Gesture"], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv("outputs/evaluation_report.csv", index=True)
    print("📄 saved: outputs/evaluation_report.csv")

    # === Confusion Matrix ===
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NoGesture", "Gesture"], yticklabels=["NoGesture", "Gesture"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - PoseViT")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300)
    print("🖼️ saved: outputs/confusion_matrix.png")

    print("\n📊 Summary Metrics:")
    print(df_report)

if __name__ == "__main__":
    evaluate(
        model_path="outputs/ckpts/posevit_best.pt",
        index_csv="data/index_test.csv",  # 테스트용 인덱스
        d_in=80,
        device="cuda:0"
    )
