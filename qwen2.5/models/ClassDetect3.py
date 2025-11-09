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
		    "text": "You are a professional hand gesture classification expert. Your task is to observe a video and classify the dominant hand gesture being performed into one of eight predefined categories. Each category is defined below with a dictionary-style definition, followed by realistic examples.\n\nYou must return only one word — the name of the selected category — with no explanation or additional output.\n\nHere are the gesture categories:\n\n1. representing\nDefinition: A gesture that visually depicts the shape, size, position, or characteristic of an object, concept, or scene.\nExamples:\n- Moving hands apart to show the width of an object.\n- Using hand shape to imitate a mountain or triangle.\n- Tracing the curve of an imagined path while describing it.\n\n2. molding\nDefinition: A gesture that mimics physically shaping, forming, or manipulating an object with the hands.\nExamples:\n- Forming a ball shape with both hands as if holding clay.\n- Twisting the hands as if wringing out a towel.\n- Flattening palms together as if pressing dough.\n\n3. indexing\nDefinition: A gesture that points to or indicates a specific direction, object, location, or person.\nExamples:\n- Pointing left or right while giving directions.\n- Pointing to oneself or another person when referring to them.\n- Pointing upward to indicate a place above.\n\n4. drawing\nDefinition: A gesture that traces shapes, letters, symbols, or figures in the air.\nExamples:\n- Drawing a heart with the finger.\n- Sketching the letter “A” in midair.\n- Tracing the outline of a box or circle with a fingertip.\n\n5. beat\nDefinition: A simple, rhythmic gesture that aligns with the flow or stress of speech, without semantic meaning.\nExamples:\n- Light vertical hand movements while speaking.\n- Tapping fingers on the table to emphasize phrases.\n- Subtle flicks of the hand to mark rhythm in speech.\n\n6. emblematic\nDefinition: A culturally standardized gesture with a fixed, commonly recognized meaning.\nExamples:\n- Thumbs up to mean “good” or “OK.”\n- Peace sign (two fingers raised in a V-shape).\n- Hand wave to say hello or goodbye.\n\n7. acting\nDefinition: A gesture that enacts or mimics a physical action, behavior, or motor activity.\nExamples:\n- Pretending to drive a car using both hands.\n- Mimicking the action of drinking from a cup.\n- Swinging an imaginary bat or tennis racket.\n\n8. other\nDefinition: A gesture that does not clearly belong to any of the defined categories or has ambiguous function.\nExamples:\n- Fidgeting with fingers or tapping without clear intent.\n- Unstructured hand movement not related to speech.\n- Accidental or nervous movements.\n\n---\n\nClassification Instructions:\n- Observe the video and identify the most prominent hand gesture.\n- Use visual cues primarily. Use speech or audio only as supplementary context.\n- Output only the exact label name (e.g., indexing). Do not add any extra text or explanation.\n\n---\n\nExample outputs:\n- A person repeatedly taps their hand in rhythm while talking → Output: beat\n- A person draws a circle in the air while describing a layout → Output: drawing\n- A person pretends to drink from an invisible cup → Output: acting\n\nNow wait for the video input. Once it is provided, analyze it and respond with only the label name that best classifies the gesture."
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
