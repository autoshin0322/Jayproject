import os
import csv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

INPUT_DIR = "video42"
OUTPUT_CSV = "Video42Output.csv"

# CSV 헤더 생성
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "filename", "label"])

# 영상 파일 리스트
video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(".mp4")]

for idx, file_name in enumerate(video_files, start=1):
    video_path = os.path.join(INPUT_DIR, file_name)
    print(f"🎥 Processing ({idx}/{len(video_files)}): {file_name}")

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
                    "text": ("You are a professional gesture recognition analyst specialized in interpreting and labeling hand gestures from video input. Your task is to observe a given video clip and classify the most prominent hand gesture performed in it, based on its communicative function. The gesture must be classified into one of the following 8 categories:\n\n1. representing – gestures that visually depict objects, scenes, or abstract concepts.\n2. molding – gestures that mimic the shaping or transformation of objects.\n3. indexing – gestures that point or indicate a direction, object, person, or location.\n4. drawing – gestures that trace symbols, shapes, or letters in the air.\n5. beat – simple rhythmic movements that align with speech or emphasize timing, with no semantic meaning.\n6. emblematic – culturally fixed gestures that carry a known, standalone meaning (e.g., thumbs up, peace sign).\n7. acting – gestures that perform or mimic actions as if enacting a scene or motion.\n8. other – gestures that do not clearly belong to any of the categories above.\n\nYour classification should be based primarily on visual motion and hand position. You may also consider audio or transcripts if available, but only to support — not override — your visual interpretation.\n\n## Output Rules:\nOnly return the classification label, using one of the 8 categories above. Do not include any explanation, punctuation, or extra text. Output must consist of only a single word: the selected label.\n\n## Example:\nInput video: A person repeatedly taps the air with their hand in sync with speech.\nOutput:\nbeat\n\nOnce the video is provided, analyze it and return your classification result. Take a deep breath and work this out step by step to ensure an accurate answer. Output only the label word."
                    ),
                },
            ],
        }
    ]

    # 입력 준비
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print(f"🧠 Output: {output_text}")

    # CSV 저장 (index 포함)
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([idx, file_name, output_text])

print("\n✅ 모든 영상 처리 완료!")
print(f"결과 저장 경로: {OUTPUT_CSV}")
