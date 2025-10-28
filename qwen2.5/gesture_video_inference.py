import cv2
import os
import base64
from models.Qwen_V2_5 import Qwen2_5_VL_7B_Instruct
from prompt import LLMPrompt, PromptMessage

# === 비디오에서 프레임을 base64로 추출 ===
def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(img_base64)
        count += 1
    cap.release()
    return frames

# === 모델 불러오기 ===
model = Qwen2_5_VL_7B_Instruct(version="main")

# === 비디오 파일 설정 ===
video_path = "005.mp4"  # 현재 디렉토리에 있는 비디오

# === 프레임 추출 ===
print("🎞️ 비디오에서 프레임 추출 중...")
frames_base64 = extract_frames(video_path, frame_rate=32)  # 1초에 1프레임

print(f"✅ {len(frames_base64)}개의 프레임 추출 완료")

# === 질문 설정 ===
prompt = LLMPrompt(messages=[
    PromptMessage(role="user", content="이 영상에서 어떤 제스처가 나타나고 있나요?")
])

# === 추론 실행 ===
print("🧠 Qwen2.5-VL 모델 추론 중...")
result = model.process_video_frames(prompt, frames=frames_base64)

# === 결과 출력 ===
print("\n📋 제스처 인식 결과:")
print(result.meta)