import os
import csv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# -----------------------------------------------------------
# 1. 모델 및 프로세서 로드
# -----------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# -----------------------------------------------------------
# 2. 입력 / 출력 경로
# -----------------------------------------------------------
INPUT_DIR = "video42"             # 모든 .mp4 파일이 들어있는 폴더
OUTPUT_CSV = "Video42Output.csv"      # 결과 저장 파일

# -----------------------------------------------------------
# 3. CSV 헤더 생성
# -----------------------------------------------------------
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "filename", "label"])

# -----------------------------------------------------------
# 4. 영상 파일 반복 처리 (최대 10개)
# -----------------------------------------------------------
#video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".mp4")]
#video_files = video_files[:10]  # ✅ 앞의 10개 파일만 선택

#for file_name in video_files:
#    video_path = os.path.join(INPUT_DIR, file_name)
#    print(f"🎥 Processing: {file_name}")

for file_name in sorted(os.listdir(INPUT_DIR)):
    if not file_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(INPUT_DIR, file_name)
    print(f"🎥 Processing: {file_name}")

    # 메시지 구성 (원본 구조 유지)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 32.0,
                },
                {
                    "type": "text",
                    "text": ("You are an expert in hand gesture recognition and classification based on visual input. Analyze the given video (including audio and transcript if available), and classify the primary hand gesture performed in the video into one of the following eight categories: representing, molding, indexing, drawing, other, beat, emblematic, or acting. Each label is defined as follows: representing: gestures that describe an object, shape, or scene / molding: gestures that simulate shaping or transforming objects with the hands / indexing: gestures that point to a direction, object, or place / drawing: gestures that mimic drawing in the air / other: gestures that don’t clearly fit into any of the categories / beat: rhythmic gestures that follow the flow of speech without semantic meaning / emblematic: culturally defined gestures with fixed meanings (e.g., thumbs up, peace sign) / acting: gestures that mime an action or movement. Make your decision based on the visible hand movement; use speech content only to support your interpretation when necessary. Be precise and avoid ambiguity. Output must follow this rule: if I request explanation, output the label followed by a 1–2 sentence reason explaining your decision; if I request no explanation, output only the classification label. Classify this gesture with no explanation. Take a deep breath and let’s work this out in a step-by-step way to make sure we get the right answer."
                    ),
                },
            ],
        }
    ]

    # 인퍼런스 입력 준비
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 모델 실행
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    print(f"🧠 Model Output: {output_text}")

    # CSV에 저장
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([file_name, output_text])

print("\n✅ 10개 영상 처리 완료!")
print(f"결과 저장 경로: {OUTPUT_CSV}")
