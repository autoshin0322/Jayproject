#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-aware extractor for EnvisionHGDetector.
- 점검: CUDA/GPU 가용성, 드라이버/장치 정보 로깅
- 시도: TF-Lite GPU delegate 사용(가능한 경우)
- 실행: retrack_gestures(input_folder, output_folder)
- 산출물: output_dir/retracked/{tracked_videos, tracked_csv, tracked_features}/*
- 가속 옵션: 프레임 다운샘플링(--fps), 긴 영상 자동 잘라 처리(--chunk-sec)

주의:
- envisionhgdetector 내부 MediaPipe 호출이 TFLite GPU delegate를 명시적으로 받지 않으면,
  CPU(XNNPACK)로 동작할 수 있습니다. 본 스크립트는 환경 변수/사전 준비를 통해
  GPU delegate 사용 가능성을 최대화하고, 실패 시 명확히 로깅합니다.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List

# 가시적인 로그 포맷
def info(msg): print(f"[INFO] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def err(msg):  print(f"[ERROR] {msg}", file=sys.stderr)

# ---------------------------
# 0) GPU/환경 점검 유틸
# ---------------------------
def print_nvidia_smi():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,nounits,noheader"])
        lines = out.decode("utf-8").strip().splitlines()
        if lines:
            info("Detected NVIDIA GPUs:")
            for ln in lines:
                idx, name, mem_total, mem_used, util = [x.strip() for x in ln.split(",")]
                print(f"  - GPU {idx}: {name} | {mem_used}/{mem_total} MiB | util {util}%")
        else:
            warn("nvidia-smi returned no GPUs.")
    except Exception as e:
        warn(f"nvidia-smi not available or failed: {e}")

def set_cuda_visible_devices(gpu: Optional[str]):
    if gpu is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    info(f"CUDA_VISIBLE_DEVICES set to {gpu}")

def try_enable_tflite_gpu_delegate():
    """
    데스크톱 리눅스에서 TF-Lite GPU delegate를 강제로 로딩하려면
    라이브러리에서 직접 Interpreter를 구성해야 하는데,
    envisionhgdetector는 내부에서 mediapipe graph를 구성하므로
    여기서는 '환경 준비 + 힌트'만 제공합니다.
    """
    # 흔히 쓰이는 delegate so 탐색(시스템에 따라 경로가 다름)
    candidate_names = [
        "libtensorflowlite_gpu_delegate.so",
        "libtensorflowlite_flex_delegate.so",
    ]
    ok = False
    for name in candidate_names:
        for search_dir in ["/usr/lib", "/usr/local/lib"] + sys.path:
            p = Path(search_dir) / name
            if p.exists():
                info(f"Found TF-Lite GPU delegate candidate: {p}")
                ok = True
    if not ok:
        warn("TF-Lite GPU delegate .so not found on common paths. "
             "Mediapipe may fall back to CPU (XNNPACK). "
             "This is expected for many desktop installs.")

    # 힌트형 환경변수(일부 배포판/커스텀 파이프라인에서 읽음)
    os.environ.setdefault("TF_DELEGATE", "gpu")       # 일부 코드가 읽는 경우가 있음
    os.environ.setdefault("TF_TFLITE_ENABLE_GPU", "1")
    info("Set TF_DELEGATE=gpu, TF_TFLITE_ENABLE_GPU=1 (best-effort hint).")

def print_tf_runtime_hint():
    # 현재 파이썬 환경에서 TF가 어느 백엔드를 보고 있는지 안내(정보 표시용)
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            info(f"TensorFlow sees {len(gpus)} GPU(s): {[d.name for d in gpus]}")
        else:
            warn("TensorFlow sees no GPU devices — TFLite may run on CPU (XNNPACK).")
    except Exception as e:
        warn(f"TensorFlow import check failed (this is fine if TFLite-only path): {e}")

# ---------------------------
# 1) 전처리(선택): 프레임 다운샘플링 / 청크 분할
# ---------------------------
def downsample_video(src: Path, dst: Path, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"fps={fps}",
        "-an",  # 오디오 제거
        str(dst),
    ]
    info(f"Downsampling to {fps} FPS: {src.name} -> {dst.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def chunk_video(src: Path, out_dir: Path, chunk_sec: int) -> List[Path]:
    """
    긴 영상을 chunk_sec 단위로 쪼개서 반환.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # 먼저 길이 확인
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", str(src)
    ]).decode("utf-8").strip()
    duration = float(probe)
    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_sec, duration)
        out = out_dir / f"{src.stem}_part{idx:02d}.mp4"
        cmd = ["ffmpeg", "-y", "-ss", f"{start}", "-to", f"{end}", "-i", str(src),
               "-c", "copy", "-an", str(out)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        chunks.append(out)
        idx += 1
        start = end
    info(f"Chunked {src.name} into {len(chunks)} parts.")
    return chunks

# ---------------------------
# 2) Envision 실행
# ---------------------------
def run_envision_retrack(input_folder: Path, output_folder: Path, model_type: str) -> None:
    from envisionhgdetector import GestureDetector

    # LightGBM 추론은 보통 CPU로도 충분히 빠름(추론 단계 GPU 이득 제한적)
    detector = GestureDetector(
        model_type=model_type,   # "lightgbm" or "cnn"
        motion_threshold=0.5,
        gesture_threshold=0.6,
        min_gap_s=0.3,
        min_length_s=0.5,
        gesture_class_bias=0.0
    )
    info(f"Running retrack_gestures: {input_folder} -> {output_folder}")
    detector.retrack_gestures(str(input_folder), str(output_folder))
    info("retrack_gestures() finished.")

def has_results(output_folder: Path) -> bool:
    rdir = output_folder / "retracked"
    if not rdir.exists():
        return False
    for sub in ["tracked_features", "tracked_csv", "tracked_videos"]:
        if any((rdir / sub).glob("*")):
            return True
    return False

# ---------------------------
# 3) 메인
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="GPU-aware Envision extractor")
    ap.add_argument("--input", required=True, help="Input folder with .mp4 (videos_to_label)")
    ap.add_argument("--output", required=True, help="Output folder (e.g., output_envision)")
    ap.add_argument("--model_type", default="lightgbm", choices=["lightgbm", "cnn"])
    ap.add_argument("--gpu", default=None, help="GPU index to expose (e.g., 0). Leave empty to keep system default.")
    ap.add_argument("--fps", type=int, default=None, help="Downsample videos to this FPS before processing (optional)")
    ap.add_argument("--chunk-sec", type=int, default=None, help="Split long videos into N-second chunks before processing (optional)")
    ap.add_argument("--workdir", default=".envision_tmp", help="Working folder for downsample/chunk output")
    args = ap.parse_args()

    in_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    work_dir = Path(args.workdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 환경/장치 로그
    print_nvidia_smi()
    set_cuda_visible_devices(args.gpu)
    try_enable_tflite_gpu_delegate()
    print_tf_runtime_hint()

    # 입력 수집
    videos = sorted([p for p in in_dir.glob("*.mp4")])
    if not videos:
        err(f"No .mp4 found in {in_dir}")
        sys.exit(1)
    info(f"Found {len(videos)} video(s) in {in_dir}")

    # 전처리(선택) — 다운샘플/청크
    prep_dir = work_dir / "prep_videos"
    prep_dir.mkdir(parents=True, exist_ok=True)
    prepared: List[Path] = []

    for v in videos:
        candidate = v
        # 다운샘플
        if args.fps:
            ds_out = prep_dir / f"{v.stem}_fps{args.fps}.mp4"
            if not ds_out.exists():
                try:
                    downsample_video(v, ds_out, args.fps)
                except subprocess.CalledProcessError:
                    warn(f"Downsample failed for {v.name}, fallback to original.")
                    ds_out = v
            candidate = ds_out

        # 청크 분할
        if args.chunk_sec:
            try:
                parts = chunk_video(candidate, prep_dir / f"{v.stem}_chunks", args.chunk_sec)
                prepared.extend(parts)
                continue
            except subprocess.CalledProcessError:
                warn(f"Chunking failed for {candidate.name}, use single file instead.")
                prepared.append(candidate)
        else:
            prepared.append(candidate)

    # 처리 전 출력 안내
    info(f"Prepared {len(prepared)} file(s) for Envision.")

    # Envision 실행은 폴더 단위이므로, 준비된 파일들을 임시 입력폴더에 모아 한 번 호출
    staged_dir = work_dir / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    # 심볼릭 링크(또는 복사)로 모으기
    for src in prepared:
        dst = staged_dir / src.name
        if dst.exists():
            dst.unlink()
        try:
            os.symlink(src, dst)
        except Exception:
            # 심볼릭 링크가 안 되면 복사
            import shutil
            shutil.copy2(src, dst)

    # Envision 실행
    run_envision_retrack(staged_dir, out_dir, args.model_type)

    # 결과 확인/요약
    if has_results(out_dir):
        info(f"✅ Done. Results under: {out_dir}/retracked/")
        # 인덱스 요약 파일이 있으면 한 줄 출력
        idx = out_dir / "index.csv"
        if idx.exists():
            info(f"Index file: {idx}")
    else:
        warn("No retracked outputs found. If this was the first run, "
             "model downloads + CPU fallback could make it slow. "
             "Check logs above and consider --fps / --chunk-sec options.")

if __name__ == "__main__":
    main()