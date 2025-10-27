import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import random

from envisionhgdetector import GestureDetector

# --- 1. 경로 및 상수 설정 ---
BASE_PATH = ""  # (가정) CSV와 비디오가 있는 기본 폴더
GROUND_TRUTH_CSV = os.path.join(BASE_PATH, "annotations.csv")
INPUT_VIDEO_FILE = os.path.join(BASE_PATH, "videos_to_label/input.mp4")
OUTPUT_PATH = os.path.join(BASE_PATH, "output") # 임시 파일 저장 경로

TEMP_PRED_CSV = os.path.join(OUTPUT_PATH, "input.mp4_segments.csv")

FPS = 32
timestep = 1 / FPS
START_TIME = 608 # 분석 시작 시점 (초)
END_TIME = 1472    # 분석 종료 시점 (초)

def run_detector(config, video_file_path, output_csv_path):
    print(f"Running detector with config: {config}")
    try:
        detector=GestureDetector(
        model_type="lightgbm", motion_threshold=config.motion_threshold, gesture_threshold=config.gesture_threshold, min_gap_s=config.min_gap_s, min_length_s=config.min_length_s)
        detector.process_folder(video_file_path, output_csv_path)
        print("--- (가상 실행 중) Detector가 예측 CSV를 생성했다고 가정합니다. ---")
        dummy_data = {'start_time': [random.uniform(0, 50), random.uniform(60, 100)], 
                      'end_time': [random.uniform(51, 59), random.uniform(101, 120)], 
                      'labelid': [1, 2], 'label': ['fist', 'palm'], 'duration': [8.5, 40.5]}
        pd.DataFrame(dummy_data).to_csv(output_csv_path, index=False)
        print(f"--- (가상 실행 완료) 임시 CSV 저장: {output_csv_path} ---")

        return True
    
    except Exception as e:
        print(f"Error running detector: {e}")
        return False

def evaluate_and_log(prediction_path, ground_truth_path, config):
    """
    두 CSV를 프레임(timestep) 단위로 비교하고 W&B에 결과를 로그합니다.
    (사용자 `eval.py` 코드 기반으로 작성)
    """
    print("Starting evaluation...")
    try:
        true_df = pd.read_csv(ground_truth_path)
        pred_df = pd.read_csv(prediction_path)

        # === ⏱️ 시간 보정 (사용자 코드 로직) ===
        # (Detector가 0초부터 시작하는 시간을 출력하고,
        #  평가는 START_TIME부터 시작하는 전역 시간 기준이므로 보정)
        pred_df["start_time"] = pred_df["start_time"] + START_TIME
        pred_df["end_time"] = pred_df["end_time"] + START_TIME

        # === 전체 시간 구간 생성 (사용자 코드 로직) ===
        timeline = np.arange(START_TIME, END_TIME, timestep)
        y_true = np.zeros(len(timeline))
        y_pred = np.zeros(len(timeline))

        # === Ground Truth (사용자 코드 로직) ===
        # (CSV의 'label'에 상관없이 모든 제스처를 '1'로 처리. 즉, 이진 분류)
        for _, row in true_df.iterrows():
            start, end = row["start_time"], row["end_time"]
            if end < START_TIME or start > END_TIME: continue
            mask = (timeline >= start) & (timeline <= end)
            y_true[mask] = 1

        # === Model Prediction (사용자 코드 로직) ===
        for _, row in pred_df.iterrows():
            start, end = row["start_time"], row["end_time"]
            if end < START_TIME or start > END_TIME: continue
            mask = (timeline >= start) & (timeline <= end)
            y_pred[mask] = 1

        # === 📊 성능 평가 (사용자 코드 로직) ===
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred)
        }
        
        print(f"--- Metrics ---")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1_score']:.4f}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")

        # === (W&B) W&B에 숫자 메트릭 로그 ===
        wandb.log(metrics)

        # === (W&B) Confusion Matrix 생성 및 로그 ===
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        cm_path = os.path.join(OUTPUT_PATH, f"confusion_matrix_{wandb.run.id}.png")
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["True_NoGesture", "True_Gesture"], columns=["Pred_NoGesture", "Pred_Gesture"])
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues")
        plt.title(f"Confusion Matrix (f1={metrics['f1_score']:.3f})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        
        # W&B 대시보드에서 바로 확인 가능하도록 이미지도 로그
        wandb.log({"confusion_matrix": wandb.Image(cm_path)})
        print(f"🖼️ Confusion Matrix saved and logged.")
        
        return True

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False

# === 4. (W&B) 메인 실행 함수 ===
def main():
    """W&B Sweep 에이전트가 호출할 메인 함수"""
    run = None
    try:
        # 1. W&B 초기화 (W&B Sweep이 config를 자동으로 주입)
        run = wandb.init()
        config = wandb.config  # W&B Sweep이 정해준 파라미터 조합
        
        # 2. Detector 실행 -> TEMP_PRED_CSV 생성
        success_run = run_detector(config, INPUT_VIDEO_FILE, TEMP_PRED_CSV)

        if not success_run:
            raise Exception("Detector failed to run.")

        # 3. 평가 실행 (사용자 코드) -> 메트릭 계산 및 W&B 로그
        evaluate_and_log(TEMP_PRED_CSV, GROUND_TRUTH_CSV, config)
        
        print(f"--- Run {run.id} finished successfully. ---")

    except Exception as e:
        print(f"An error occurred in main function: {e}")
    
    finally:
        # 임시 파일 정리
        if os.path.exists(TEMP_PRED_CSV):
            os.remove(TEMP_PRED_CSV)
        if run:
            run.finish() # W&B 실행 종료

# --- 스크립트 시작점 ---
if __name__ == "__main__":
    main()
