import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn
import gc
import os
import tempfile
from moviepy import VideoFileClip
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """Load the model and processor."""
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

def unload_models():
    """Unload the model and processor to free memory."""
    global model, processor
    try:
        if model is not None:
            # Move model to CPU first to ensure all GPU tensors are freed
            model = model.cpu()
            # Delete model and clear CUDA cache
            del model
            model = None
            
        if processor is not None:
            del processor
            processor = None
            
        if torch.cuda.is_available():
            # Reset CUDA memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Synchronize CUDA to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Get current memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
            print(f"After cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
    except Exception as e:
        print(f"Error during model unloading: {str(e)}")
        raise e

@app.post("/load_model")
async def load_model():
    """Load the model and processor."""
    try:
        load_models()
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload_model")
async def unload_model():
    """Unload the model and processor."""
    try:
        unload_models()
        return {"status": "success", "message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

def create_temp_video_clip(video_path: str, start_time: Optional[float], end_time: Optional[float]) -> str:
    """Create a temporary video clip from the original video using the specified timestamps."""
    try:
        # Get the original file extension
        _, ext = os.path.splitext(video_path)
        if not ext:
            ext = '.mp4'  # Default to .mp4 if no extension found
            
        # Create a temporary file with the same extension
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_path = temp_file.name
        temp_file.close()

        logger.debug(f"Creating temporary video clip from {video_path}")
        logger.debug(f"Start time: {start_time}, End time: {end_time}")

        # Load the video
        video = VideoFileClip(video_path)
        logger.debug(f"Original video duration: {video.duration}")
        
        # If timestamps are provided, create a subclip
        if start_time is not None or end_time is not None:
            start = start_time if start_time is not None else 0
            end = end_time if end_time is not None else video.duration
            logger.debug(f"Creating subclip from {start} to {end}")
            video = video.subclipped(start, end)
        
        # Write the video to the temporary file
        logger.debug(f"Writing video to temporary file: {temp_path}")
        video.write_videofile(temp_path, codec='libx264', audio_codec='aac')
        video.close()
        
        return temp_path
    except Exception as e:
        error_msg = f"Error creating video clip: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def cleanup_temp_video(temp_path: str):
    """Clean up the temporary video file."""
    try:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug(f"Cleaned up temporary file: {temp_path}")
    except Exception as e:
        logger.warning(f"Error cleaning up temporary file {temp_path}: {str(e)}")

@app.post("/generate_caption")
async def generate_caption(request: CaptionRequest):
    temp_video_path = None
    try:
        logger.debug(f"Received caption request for video: {request.video_path}")
        logger.debug(f"Request parameters: {request.dict()}")

        if model is None or processor is None:
            error_msg = "Model not loaded. Please load the model first."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
            
        video_path = request.video_path.replace("/home/broliang", "/data/shared/Qwen")
        logger.debug(f"Processed video path: {video_path}")

        # Create temporary video clip if timestamps are provided
        if request.start_time is not None or request.end_time is not None:
            temp_video_path = create_temp_video_clip(
                video_path,
                request.start_time,
                request.end_time
            )
            video_path_to_use = temp_video_path
            logger.debug(f"Using temporary video clip: {temp_video_path}")
        else:
            video_path_to_use = video_path
            logger.debug("Using original video path")

        # Prepare the message with video information
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path_to_use,
                        "max_pixels": request.max_pixels,
                        "fps": request.fps,
                    },
                    {"type": "text", "text": request.query},
                ],
            }
        ]

        # Add transcript context if provided
        if request.transcript:
            messages[0]["content"][1]["text"] = f"Given the transcript: '{request.transcript}', {request.query}"
        
        logger.debug(f"Prepared messages for model: {messages}")

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
        logger.debug("Generating caption...")
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.debug(f"Generated caption: {output_text}")
        return {"caption": output_text}

    except Exception as e:
        error_msg = f"Error generating caption: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Clean up temporary video file if it was created
        if temp_video_path:
            cleanup_temp_video(temp_video_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)