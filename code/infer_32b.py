import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "./models/Qwen2.5-VL-32B-Instruct-AWQ", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "./models/Qwen2.5-VL-32B-Instruct",
    # torch_dtype=torch.bfloat16,
    # torch_dtype="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    # load_in_8bit=True,
    load_in_4bit=True,
)

# default processer
processor = AutoProcessor.from_pretrained("./models/Qwen2.5-VL-32B-Instruct")

fps = 5.0
# Messages containing a images list as a video and a text query
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": [
#                     "file:///path/to/frame1.jpg",
#                     "file:///path/to/frame2.jpg",
#                     "file:///path/to/frame3.jpg",
#                     "file:///path/to/frame4.jpg",
#                 ],
#             },
#             {"type": "text", "text": "Describe this video."},
#         ],
#     }
# ]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "/data/shared/Qwen/videos/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image in detail."},
#         ],
#     }
# ]
# Messages containing a local video path and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/data/shared/Qwen/videos/videoplayback.mp4",
                "max_pixels": 360 * 420,
                "fps": fps,
            },
            # {"type": "text", "text": "Is there a scene inside a car? If so, what is the start and end timestamp of this scene? what's the color of the car's internal decoration?"},
            {"type": "text", "text": "Is there a scene inside an office? If so, what is the start and end timestamp of this scene?"},
            # {"type": "text", "text": "For each frame, describe the frame in detail, focusing on the scene objects and characters, don't infer or make guesses of the story"},
            # {"type": "text", "text": "Who is bullied, the girl or her classmates?"},
        ],
    }
]

# Messages containing a video url and a text query
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
#             },
#             {"type": "text", "text": "Describe this video."},
#         ],
#     }
# ]

#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    # fps=fps,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda", dtype=torch.float16)
# inputs = inputs.to("cuda", dtype=torch.bfloat16)
# inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)