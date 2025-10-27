import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os

# [!!!] envisionhgdetector 임포트
from envisionhgdetector import GestureDetector 

# --- 1. 경로 및 상수 설정 ---
BASE_PATH = ""  # CSV와 비디오가 있는 기본 폴더
GROUND_TRUTH_CSV = os.path.join(BASE_PATH, "annotations.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "output") # 임시 파일 저장 경로

# [!!!] Detector가 실행할 비디오가 *포함된* 폴더
# (process_folder는 폴더를 입력으로 받습니다)
INPUT_VIDEO_FOLDER = os.path.join(BASE_PATH, "videos_to_label")

# [!!!] Detector가 생성할 예측 파일의 *정확한* 경로
# (사용자의 원본 eval.py를 참고하여 "input.mp4_segments.csv"로 가정)
PREDICTION_CSV_PATH = os.path.join(OUTPUT_PATH, "input.mp4_segments.csv")

# [!!!] 평가 로직에 필요한 상수 (사용자 eval.py 기반)
FPS = 32
timestep = 1 / FPS
START_TIME = 608 # 분석 시작 시점 (초)
END_TIME = 1472    # 분석 종료 시점 (초)


# --- 2. Detector 실행 함수 (가상 코드 -> 실제 코드로 교체!) ---
def run_detector(config, video_folder_path, output_folder_path):
    """
    W&B config로 Detector를 실행하고 예측 CSV를 저장합니다.
    """
    print(f"Running detector with config: {config}")
    try:
        # 1. W&B config에서 파라미터를 가져와 Detector 초기화
        # (원본 코드의 하드코딩된 값을 W&B config 값으로 대체)
        detector = GestureDetector(
            model_type="lightgbm",
            motion_threshold=config.motion_threshold, # W&B 값 사용
            gesture_threshold=config.gesture_threshold, # W&B 값 사용
            min_gap_s=config.min_gap_s,                 # W&B 값 사용
            min_length_s=config.min_length_s            # W&B 값 사용
        )

        # 2. (핵심) 제공해주신 실제 코드를 실행
        # (dummy_data 코드는 모두 삭제되었습니다)
        print(f"Processing folder: {video_folder_path} -> {output_folder_path}")
        detector.process_folder(
            input_folder=video_folder_path,
            output_folder=output_folder_path,
        )
        print(f"Detector processing finished.")
        
        # 3. Detector가 예측 파일을 생성했는지 확인
        if not os.path.exists(PREDICTION_CSV_PATH):
            raise FileNotFoundError(f"Detector did not create the expected file: {PREDICTION_CSV_PATH}")

        return True
    
    except Exception as e:
        print(f"Error running detector: {e}")
        return False

# --- 3. 평가 및 로그 함수 (수정 불필요) ---
def evaluate_and_log(prediction_path, ground_truth_path, config):
    """
    두 CSV를 프레임(timestep) 단위로 비교하고 W&B에 결과를 로그합니다.
    (사용자 `eval.py` 코드 기반으로 작성)
    """
    print(f"Evaluating: {prediction_path} vs {ground_truth_path}")
    try:
        true_df = pd.read_csv(ground_truth_path)
        pred_df = pd.read_csv(prediction_path)

        # === ⏱️ 시간 보정 (사용자 코드 로직) ===
        pred_df["start_time"] = pred_df["start_time"] + START_TIME
        pred_df["end_time"] = pred_df["end_time"] + START_TIME

        # === 전체 시간 구간 생성 (사용자 코드 로직) ===
        timeline = np.arange(START_TIME, END_TIME, timestep)
        y_true = np.zeros(len(timeline))
        y_pred = np.zeros(len(timeline))

        # === Ground Truth (사용자 코드 로직) ===
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
        
        wandb.log({"confusion_matrix": wandb.Image(cm_path)})
        print(f"🖼️ Confusion Matrixsaved and logged.")
        
        return True

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False

# --- 4. (W&B) 메인 실행 함수 ---
def main():
    """W&B Sweep 에이전트가 호출할 메인 함수"""
    run = None
    try:
        # 1. W&B 초기화 (W&B Sweep이 config를 자동으로 주입)
        run = wandb.init()
        config = wandb.config  # W&B Sweep이 정해준 파라미터 조합
        
        # 2. Detector 실행 -> PREDICTION_CSV_PATH 파일 생성
        success_run = run_detector(config, INPUT_VIDEO_FOLDER, OUTPUT_PATH)

        if not success_run:
            raise Exception("Detector failed to run or did not create output file.")

        # 3. 평가 실행 (사용자 코드) -> 메트릭 계산 및 W&B 로그
        evaluate_and_log(PREDICTION_CSV_PATH, GROUND_TRUTH_CSV, config)
        
        print(f"--- Run {run.id} finished successfully. ---")

    except Exception as e:
        print(f"An error occurred in main function: {e}")
    
    finally:
        # 임시 예측 파일 정리 (다음 실행을 위해)
        if os.path.exists(PREDICTION_CSV_PATH):
            os.remove(PREDICTION_CSV_PATH)
        if run:
            run.finish() # W&B 실행 종료

# --- 스크립트 시작점 ---
if __name__ == "__main__":
    main()