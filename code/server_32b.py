import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn

app = FastAPI()

# Global variables for model and processor
model = None
processor = None
MODEL_PATH = "./models/Qwen2.5-VL-32B-Instruct"

class CaptionRequest(BaseModel):
    video_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    transcript: Optional[str] = None
    query: str
    fps: float = 5.0
    max_pixels: int = 360 * 420

def load_models():
    global model, processor
    if model is None:
        # We recommend enabling flash_attention_2 for better acceleration and memory saving
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            load_in_4bit=True,
        )
    if processor is None:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        if model is None:
            return {
                "status": "Model not loaded",
                "model_path": MODEL_PATH,
                "model_type": "Qwen2.5-VL-32B-Instruct"
            }
        
        return {
            "status": "Model loaded",
            "model_path": MODEL_PATH,
            "model_type": "Qwen2.5-VL-32B-Instruct",
            "model_config": {
                "dtype": str(model.dtype),
                "device": str(next(model.parameters()).device),
                "quantization": "4-bit" if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit else "None",
                "flash_attention": "Enabled" if hasattr(model, "config") and getattr(model.config, "use_flash_attention_2", False) else "Disabled"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_caption")
async def generate_caption(request: CaptionRequest):
    try:
        # Prepare the message with video information
        video_path = request.video_path.replace("/home/broliang", "/data/shared/Qwen")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": request.max_pixels,
                        "fps": request.fps,
                        "start_time": request.start_time,
                        "end_time": request.end_time,
                    },
                    {"type": "text", "text": request.query},
                ],
            }
        ]

        # Add transcript context if provided
        if request.transcript:
            messages[0]["content"][1]["text"] = f"Given the transcript: '{request.transcript}', {request.query}"
        
        print(messages)

        # Process the input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda", dtype=torch.float16)

        # Generate caption
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(output_text)

        return {"caption": output_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)