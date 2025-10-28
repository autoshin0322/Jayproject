# run_envision_gpu.py
import os, sys, time, glob
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Mediapipe는 Python wheel 기준 GPU delegate가 거의 불가 → 대신 라이트 모델 사용
os.environ["MEDIAPIPE_USE_LITE_MODEL"] = "1"  # 커스텀 키 (아래에서 참고)
# TFLite GPU delegate 힌트(실제 적용 안 될 수 있으나 무해)
os.environ["TF_TFLITE_ENABLE_GPU"] = "1"

print("▶ TensorFlow:", tf.__version__)
print("▶ GPUs:", tf.config.list_physical_devices('GPU'))

from envisionhgdetector import GestureDetector, utils

def main(input_folder="videos_to_label", output_folder="output_envision",
         model_type="lightgbm", fps_target=16, save_tracked_video=False):
    os.makedirs(output_folder, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
    if not videos:
        print(f"❌ No videos in {input_folder}")
        return

    # Envision 내부는 mediapipe heavy가 기본일 수 있어, features 추출만 빠르게 진행
    # CNN을 쓸 경우 TF는 GPU로 가속됨, LightGBM은 CPU이지만 빠릅니다.
    detector = GestureDetector(
        model_type=model_type,
        motion_threshold=0.5,
        gesture_threshold=0.6,
        min_gap_s=0.3,
        min_length_s=0.5
    )

    print(f"▶ Found {len(videos)} video(s). Starting...")
    t0 = time.time()

    # retrack_gestures(folder_in, folder_out) 만 제공됨 → 폴더단위 처리
    # 아래 유틸로 'fps 다운샘플링 + 라이트 모델'을 강제하려면 사전 전처리가 필요할 수 있음.
    # 간단화: utils.cut_video_by_fps 로 16fps 버전 임시 생성 (없으면 패스)
    stage_dir = os.path.join(output_folder, "_staged_16fps")
    os.makedirs(stage_dir, exist_ok=True)

    # 1) FPS 다운샘플링 (가용 시)
    for v in videos:
        try:
            outv = os.path.join(stage_dir, os.path.basename(v))
            if not os.path.exists(outv):
                utils.downsample_video(v, outv, target_fps=fps_target)
        except Exception as e:
            print(f"⚠ downsample skip for {v}: {e}")

    # 2) 추적 실행 (Mediapipe는 CPU, CNN은 GPU)
    tracked_dir = os.path.join(output_folder, "retracked", "tracked_videos")
    os.makedirs(tracked_dir, exist_ok=True)
    print("▶ Running retrack_gestures (folder level)...")
    detector.retrack_gestures(stage_dir, output_folder)

    # 3) 결과 요약
    took = time.time() - t0
    print(f"✅ Done in {took/60:.1f} min")
    print("▶ Outputs:")
    print("  - CSVs :", glob.glob(os.path.join(output_folder, "*.csv")))
    print("  - Tracked videos:", glob.glob(os.path.join(tracked_dir, "*.mp4")))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="videos_to_label")
    ap.add_argument("--output", default="output_envision")
    ap.add_argument("--model_type", default="lightgbm", choices=["lightgbm","cnn"])
    ap.add_argument("--fps", type=int, default=16, help="downsample fps for pose extraction")
    args = ap.parse_args()

    # GPU 선택 (TF만 영향, Mediapipe는 CPU)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main(args.input, args.output, args.model_type, fps_target=args.fps)