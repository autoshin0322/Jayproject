from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="cuda:1"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "videos_to_label/video43-026.mp4",
                "max_pixels": 360 * 420,
                "fps": 32.0,
            },
            {"type": "text", "text": "You are an expert in hand gesture recognition and classification based on visual input. Analyze the given video (including audio and transcript if available), and classify the primary hand gesture performed in the video into one of the following eight categories: representing, molding, indexing, drawing, other, beat, emblematic, or acting. Each label is defined as follows: representing: gestures that describe an object, shape, or scene / molding: gestures that simulate shaping or transforming objects with the hands / indexing: gestures that point to a direction, object, or place / drawing: gestures that mimic drawing in the air / other: gestures that don’t clearly fit into any of the categories / beat: rhythmic gestures that follow the flow of speech without semantic meaning / emblematic: culturally defined gestures with fixed meanings (e.g., thumbs up, peace sign) / acting: gestures that mime an action or movement. Make your decision based on the visible hand movement; use speech content only to support your interpretation when necessary. Be precise and avoid ambiguity. Output should include: (1) the selected label as a single word, followed by (2) a brief explanation (1–2 sentences) describing why you chose that label based on the gesture observed. Take a deep breath and let’s work this out in a step-by-step way to make sure we get the right answer."},
        ],
    }
]
# Preparation for inference
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
)
inputs = inputs.to("cuda:1")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
