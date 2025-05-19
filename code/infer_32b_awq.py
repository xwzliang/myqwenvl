# import openai

# port = 8000
# client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
# fps = 5.0

# response = client.chat.completions.create(
#     model="/models/qwenvl/Qwen2.5-VL-32B-Instruct-AWQ",
#     messages = [
#         {
#             "role": "user",
#             "content": "Tell me a joke.",
#             # "content": [
#             #     {
#             #         "type": "video",
#             #         "video": "/videos/videoplayback.mp4",
#             #         "max_pixels": 360 * 420,
#             #         "fps": fps,
#             #     },
#             #     # {"type": "text", "text": "Is there a scene inside a car? If so, what is the start and end timestamp of this scene? what's the color of the car's internal decoration?"},
#             #     {"type": "text", "text": "Is there a scene inside an office? If so, what is the start and end timestamp of this scene?"},
#             #     # {"type": "text", "text": "For each frame, describe the frame in detail, focusing on the scene objects and characters, don't infer or make guesses of the story"},
#             #     # {"type": "text", "text": "Who is bullied, the girl or her classmates?"},
#             # ],
#         }
#     ],
#     temperature=0,
#     max_tokens=512,
# )

# print(f"Response: {response}")

import cv2

def extract_video_frames(video_path, max_frames=3, resize=(224, 224)):
    """
    Extracts up to `max_frames` evenly spaced frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to extract.
        resize (tuple): Resize frame to this shape (width, height).
    
    Returns:
        List of np.ndarray: Frames as NumPy arrays in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"Cannot read video: {video_path}")

    frame_indices = [
        int(i * total_frames / max_frames) for i in range(max_frames)
    ]

    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if idx in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame_rgb = cv2.resize(frame_rgb, resize)
            frames.append(frame_rgb)

    cap.release()
    return frames


from vllm import LLM

# Initialize the model
llm = LLM(model="/models/qwenvl/Qwen2.5-VL-32B-Instruct-AWQ")

# Extract frames
video_path = "/videos/videoplayback.mp4"
video_frames = extract_video_frames(video_path)

# Prepare input
prompt = "Describe what's happening in the video."
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "video": video_frames
    }
}

# Generate response
outputs = llm.generate(inputs)
print(outputs[0].outputs[0].text)