#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback
import pandas as pd
from envisionhgdetector.detector import GestureDetector


def extract_features(input_folder, output_folder, model_type="cnn"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_folder.glob("*.mp4"))
    if not video_files:
        print(f"❌ No .mp4 files found in {input_folder}")
        return

    print(f"▶ Found {len(video_files)} videos in {input_folder}")
    print(f"▶ Extracting features to {output_folder}")

    detector = GestureDetector(model_type=model_type)

    try:
        print("▶ Running retrack_gestures on folder...")
        result_dict = detector.retrack_gestures(str(input_folder), str(output_folder))
        print("✅ retrack_gestures finished.")
        print("Returned result:", result_dict)

    except Exception as e:
        print(f"❌ retrack_gestures() failed: {e}")
        traceback.print_exc()
        return

    # === retrack_gestures 실행 후 feature 파일 탐색 ===
    all_features = list(output_folder.rglob("*feature*.npy")) + list(output_folder.rglob("*feature*.csv"))

    if not all_features:
        print("⚠️ No feature files found after retrack_gestures(). "
              "Possible reasons: failed hand tracking or missing export.")
        return

    index_records = []
    for f in all_features:
        video_name = f.stem.split("_")[0]
        index_records.append({
            "video": video_name,
            "feature_path": str(f)
        })

    df = pd.DataFrame(index_records)
    df.to_csv(output_folder / "index.csv", index=False)
    print(f"✅ Saved feature index: {output_folder/'index.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Extract pose-based features from EnvisionHGDetector retracking output.")
    parser.add_argument("--input", required=True, help="Folder with .mp4 videos")
    parser.add_argument("--output", required=True, help="Output folder for feature files")
    parser.add_argument("--model_type", default="cnn", choices=["cnn", "lightgbm"], help="Backbone for detection")
    args = parser.parse_args()

    extract_features(args.input, args.output, args.model_type)


if __name__ == "__main__":
    main()