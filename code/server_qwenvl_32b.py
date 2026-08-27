import torch
import av
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File, Form
import gc
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List, Literal
import uvicorn
from moviepy import VideoFileClip

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "server.log")
DEBUG_PAYLOAD_LOGGING = os.getenv("QWENVL_DEBUG_PAYLOADS", "").lower() in {
    "1",
    "true",
    "yes",
}
ATTENTION_BACKEND = os.getenv("QWENVL_ATTENTION_BACKEND", "flash_attention_2")
if ATTENTION_BACKEND not in {"sdpa", "flash_attention_2"}:
    raise ValueError(
        "QWENVL_ATTENTION_BACKEND must be 'sdpa' or 'flash_attention_2'"
    )

_log_file_handle = None
_log_lock = threading.Lock()


def write_log(message, level="INFO"):
    """Write a log message using one persistent, line-buffered file handle."""
    global _log_file_handle
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {level} - {message}\n"

        with _log_lock:
            print(log_message, end="")
            if _log_file_handle is None:
                _log_file_handle = open(
                    log_file, "a", encoding="utf-8", buffering=1
                )
            _log_file_handle.write(log_message)
    except Exception as e:
        print(f"Error writing to log: {str(e)}")


def write_debug_log(message):
    """Write payload-heavy diagnostics only when explicitly enabled."""
    if DEBUG_PAYLOAD_LOGGING:
        write_log(message, "DEBUG")


def rotate_log_file(log_file, max_size_mb=10):
    """Rotate log file if it exceeds the maximum size.

    Args:
        log_file (str): Path to the log file
        max_size_mb (int): Maximum size in MB before rotation
    """
    try:
        if not os.path.exists(log_file):
            return

        # Get file size in MB
        file_size_mb = os.path.getsize(log_file) / (1024 * 1024)

        if file_size_mb > max_size_mb:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{log_file}.{timestamp}"

            # Copy current log file to backup
            shutil.copy2(log_file, backup_file)

            # Empty the current log file
            with open(log_file, "w") as f:
                f.write(f"=== Log rotated at {datetime.now()} ===\n")

            write_log(f"Log file rotated. Backup created at: {backup_file}")
    except Exception as e:
        write_log(f"Error rotating log file: {str(e)}", "ERROR")


# Rotate log file if it exists
rotate_log_file(log_file)

app = FastAPI()

# Global variables for model and processor
model = None
processor = None
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Qwen3-VL-8B-Instruct")
)


def load_models():
    """Load the model and processor."""
    global model, processor
    if model is None:
        write_log("Loading model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION_BACKEND,
            device_map={"": 0},
        ).eval()
        write_log(
            f"Model loaded successfully with attention={ATTENTION_BACKEND}, "
            f"dtype={model.dtype}"
        )
    if processor is None:
        write_log("Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        processor.tokenizer.padding_side = "left"
        write_log("Tokenizer padding side set to left")
        write_log("Processor loaded successfully")


def unload_models():
    """Unload the model and processor to free memory."""
    global model, processor
    try:
        if model is not None:
            write_log("Moving model to CPU...")
            # Move model to CPU first to ensure all GPU tensors are freed
            model = model.cpu()
            # Delete model and clear CUDA cache
            del model
            model = None
            write_log("Model unloaded successfully")

        if processor is not None:
            del processor
            processor = None
            write_log("Processor unloaded successfully")

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

            torch.cuda.ipc_collect()  # Cleanup inter-process memory if using multiprocessing

            # Get current memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
            write_log(
                f"After cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )

    except Exception as e:
        error_msg = f"Error during model unloading: {str(e)}"
        write_log(error_msg, "ERROR")
        raise e


@app.post("/load_model")
async def load_model():
    """Load the model and processor."""
    try:
        load_models()
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/unload_model")
async def unload_model():
    """Unload the model and processor."""
    try:
        unload_models()
        return {"status": "success", "message": "Model unloaded successfully"}
    except Exception as e:
        error_msg = f"Error unloading model: {str(e)}"
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/self-shutdown")
async def self_shutdown():
    def exit_later():
        # short delay to ensure response is sent
        import time

        time.sleep(0.1)
        os._exit(0)  # immediate hard exit

    threading.Thread(target=exit_later, daemon=True).start()
    return {"status": "shutting down"}


@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        if model is None:
            return {
                "status": "Model not loaded",
                "model_path": MODEL_PATH,
                "model_type": "Qwen3-VL-8B-Instruct",
            }

        return {
            "status": "Model loaded",
            "model_path": MODEL_PATH,
            "model_type": "Qwen3-VL-8B-Instruct",
            "model_config": {
                "dtype": str(model.dtype),
                "device": str(next(model.parameters()).device),
                "quantization": (
                    "4-bit"
                    if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit
                    else "None"
                ),
                "flash_attention": (
                    "Enabled"
                    if getattr(
                        getattr(model, "config", None),
                        "_attn_implementation",
                        ATTENTION_BACKEND,
                    )
                    == "flash_attention_2"
                    else "Disabled"
                ),
                "attention_backend": getattr(
                    getattr(model, "config", None),
                    "_attn_implementation",
                    ATTENTION_BACKEND,
                ),
            },
        }
    except Exception as e:
        error_msg = f"Error getting model info: {str(e)}"
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)


def create_temp_video_clip(
    video_path: str, start_time: Optional[float], end_time: Optional[float]
) -> str:
    """Create a temporary video clip from the original video using the specified timestamps."""
    try:
        # Get the original file extension
        _, ext = os.path.splitext(video_path)
        if not ext:
            ext = ".mp4"  # Default to .mp4 if no extension found

        # Create a temporary file with the same extension
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_path = temp_file.name
        temp_file.close()

        write_log(f"Creating temporary video clip from {video_path}")
        write_log(f"Start time: {start_time}, End time: {end_time}")

        # Load the video
        video = VideoFileClip(video_path)
        write_log(f"Original video duration: {video.duration}")

        # If timestamps are provided, create a subclip
        if start_time is not None or end_time is not None:
            start = start_time if start_time is not None else 0
            end = end_time if end_time is not None else video.duration
            write_log(f"Creating subclip from {start} to {end}")
            video = video.subclipped(start, end)

        # Write the video to the temporary file
        write_log(f"Writing video to temporary file: {temp_path}")
        video.write_videofile(temp_path, codec="libx264", audio_codec="aac")
        video.close()

        return temp_path
    except Exception as e:
        error_msg = f"Error creating video clip: {str(e)}\n{traceback.format_exc()}"
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)


def cleanup_temp_video(temp_path: str):
    """Clean up the temporary video file."""
    try:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            write_log(f"Cleaned up temporary file: {temp_path}")
    except Exception as e:
        write_log(f"Error cleaning up temporary file {temp_path}: {str(e)}", "WARNING")


class CaptionRequest(BaseModel):
    video_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    transcript: Optional[str] = None
    query: str
    fps: float = 5.0
    max_pixels: int = 360 * 420


@app.post("/infer_path")
async def generate_caption(request: CaptionRequest):
    temp_video_path = None
    try:
        write_log(f"Received caption request for video: {request.video_path}")
        write_debug_log(f"Request parameters: {request.dict()}")

        if model is None or processor is None:
            load_models()

        video_path = request.video_path.replace("/home/broliang", "/data/shared/Qwen")
        video_path = request.video_path.replace("~/", "/data/shared/Qwen/")
        video_path = re.sub(
            r"^/mnt/omv/resources/video_summarizer(?=/|$)(.*)",
            r"/data/shared/Qwen/videos/video_summarizer\1",
            video_path,
        )
        write_log(f"Processed video path: {video_path}")

        # Create temporary video clip if timestamps are provided
        if request.start_time is not None or request.end_time is not None:
            temp_video_path = create_temp_video_clip(
                video_path, request.start_time, request.end_time
            )
            video_path_to_use = temp_video_path
            write_log(f"Using temporary video clip: {temp_video_path}")
        else:
            video_path_to_use = video_path
            write_log("Using original video path")

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
            messages[0]["content"][1][
                "text"
            ] = f"Given the transcript: '{request.transcript}', {request.query}"

        write_debug_log(f"Prepared messages for model: {messages}")

        # Process the multimodal conversation using the Qwen3-VL processor.
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "fps": request.fps,
                "max_pixels": request.max_pixels,
            },
        )
        inputs = inputs.to(model.device)

        # Generate caption
        write_log("Generating caption...")
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        write_log(f"Generated caption: {output_text}")
        return {"caption": output_text}

    except Exception as e:
        error_msg = f"Error generating caption: {str(e)}\n{traceback.format_exc()}"
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Clean up temporary video file if it was created
        if temp_video_path:
            cleanup_temp_video(temp_video_path)


@app.post("/infer")
async def infer(
    video: UploadFile = File(...),
    query: str = Form(...),
    max_pixels: int = Form(...),
    fps: int = Form(...),
    start_time: float = Form(None),
    end_time: float = Form(None),
    transcript: str = Form(None),
):
    temp_video_path = None
    uploaded_video_path = None

    try:
        # Save the uploaded file to a temporary location
        temp_dir = "/tmp"  # or use tempfile.mkdtemp() if you want a unique folder
        filename = f"{uuid.uuid4()}.mp4"
        uploaded_video_path = os.path.join(temp_dir, filename)

        with open(uploaded_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        write_log(f"Received uploaded video: {video.filename}")
        write_log(f"Saved to temporary path: {uploaded_video_path}")

        if model is None or processor is None:
            load_models()

        # Handle clipping
        if start_time is not None or end_time is not None:
            temp_video_path = create_temp_video_clip(
                uploaded_video_path, start_time, end_time
            )
            video_path_to_use = temp_video_path
            write_log(f"Using temporary video clip: {temp_video_path}")
        else:
            video_path_to_use = uploaded_video_path
            write_log("Using full uploaded video")

        # Prepare model input
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path_to_use,
                        "max_pixels": max_pixels,
                        "fps": fps,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        if transcript:
            messages[0]["content"][1][
                "text"
            ] = f"Given the transcript: '{transcript}', {query}"

        write_debug_log(f"Prepared messages for model: {messages}")

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "fps": fps,
                "max_pixels": max_pixels,
            },
        )
        inputs = inputs.to(model.device)

        write_log("Generating caption...")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        write_log(f"Generated caption: {output_text}")
        return {"caption": output_text}

    except Exception as e:
        error_msg = f"Error generating caption: {str(e)}\n{traceback.format_exc()}"
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)

    finally:
        # Clean up temporary files
        if temp_video_path and os.path.exists(temp_video_path):
            cleanup_temp_video(temp_video_path)
        if uploaded_video_path and os.path.exists(uploaded_video_path):
            os.remove(uploaded_video_path)


class ImagesRequest(BaseModel):
    image_paths: List[str]
    query: str
    transcript: Optional[str] = None
    max_new_tokens: int = 8192  # optional override


@app.post("/infer_images_path")
async def generate_caption_from_images(request: ImagesRequest):
    try:
        write_log(f"Received images request: {len(request.image_paths)} images")
        write_debug_log(f"Request parameters: {request.dict()}")

        if model is None or processor is None:
            load_models()

        if not request.image_paths:
            raise HTTPException(status_code=400, detail="image_paths cannot be empty.")

        def strip_file_uri(p: str) -> str:
            return p[7:] if p.startswith("file://") else p

        def normalize_path(p: str) -> str:
            # replicate your video path rewrites for images
            p = strip_file_uri(p)
            p = p.replace("/home/broliang", "/data/shared/Qwen")
            p = p.replace("~/", "/data/shared/Qwen/")
            p = re.sub(
                r"^/data/video_summarizer(?=/|$)(.*)",
                r"/data/shared/Qwen/videos/video_summarizer\1",
                p,
            )
            return p

        def to_file_uri(p: str) -> str:
            # ensure "file://..." for local absolute paths
            return p if p.startswith("file://") else f"file://{p}"

        # Normalize and validate all image paths
        normalized_file_uris: List[str] = []
        for raw in request.image_paths:
            np = normalize_path(raw)
            if not os.path.exists(np):
                msg = f"Image not found: {np}"
                write_log(msg, "ERROR")
                raise HTTPException(status_code=400, detail=msg)
            normalized_file_uris.append(to_file_uri(np))

        write_log(f"Processed image URIs: {normalized_file_uris}")

        # Build messages (multiple images + text query)
        user_text = request.query
        if request.transcript:
            user_text = f"Given the transcript: '{request.transcript}', {request.query}"

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": uri} for uri in normalized_file_uris]
                    + [{"type": "text", "text": user_text}]
                ),
            }
        ]

        write_debug_log(f"Prepared messages for model: {messages}")

        # Prepare inputs using the Qwen3-VL multimodal chat template.
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs = inputs.to(model.device)

        # Inference
        write_log("Generating caption for images...")
        generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        write_log(f"Generated caption: {output_text}")
        return {"caption": output_text}

    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"Error generating caption from images: {str(e)}\n{traceback.format_exc()}"
        )
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)


class BatchImagesRequest(BaseModel):
    image_paths: List[str]
    query: str
    transcript: Optional[str] = None
    max_new_tokens: int = 8192  # optional override


@app.post("/infer_batch_images_path")
async def infer_batch_images_path(request: BatchImagesRequest):
    try:
        write_log(f"Received batch images request: {len(request.image_paths)} images")
        write_debug_log(f"Request parameters: {request.dict()}")

        if model is None or processor is None:
            load_models()

        if not request.image_paths:
            raise HTTPException(status_code=400, detail="image_paths cannot be empty.")

        def strip_file_uri(p: str) -> str:
            return p[7:] if p.startswith("file://") else p

        def normalize_path(p: str) -> str:
            p = strip_file_uri(p)
            p = p.replace("/home/broliang", "/data/shared/Qwen")
            p = p.replace("~/", "/data/shared/Qwen/")
            p = re.sub(
                r"^/mnt/omv/resources/video_summarizer(?=/|$)(.*)",
                r"/data/shared/Qwen/videos/video_summarizer\1",
                p,
            )
            return p

        def to_file_uri(p: str) -> str:
            return p if p.startswith("file://") else f"file://{p}"

        # Normalize and validate all image paths
        normalized_paths = []
        file_uris = []
        for raw in request.image_paths:
            np = normalize_path(raw)
            if not os.path.exists(np):
                msg = f"Image not found: {np}"
                write_log(msg, "ERROR")
                raise HTTPException(status_code=400, detail=msg)
            normalized_paths.append(np)
            file_uris.append(to_file_uri(np))

        write_log(f"Processed image URIs for batch: {file_uris}")

        # Common user text (optionally include transcript context)
        user_text = request.query
        if request.transcript:
            user_text = f"Given the transcript: '{request.transcript}', {request.query}"

        # Build one conversation per image
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": uri},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            for uri in file_uris
        ]

        write_log(f"Prepared {len(conversations)} conversations for batch inference")

        # Prepare batched inputs using the Qwen3-VL multimodal chat template.
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs = inputs.to(model.device)

        # Batch inference
        write_log("Running batch generation for images...")
        generated_ids = model.generate(
            **inputs, max_new_tokens=request.max_new_tokens, top_k=5
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Package results (path + caption)
        results = []
        for file_uri, caption in zip(file_uris, output_texts):
            results.append({"image": file_uri, "caption": caption})
            write_log(file_uri)
            write_log(caption)

        write_log(f"Batch generated {len(results)} captions")
        return {"caption": results}

    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"Error in batch image inference: {str(e)}\n{traceback.format_exc()}"
        )
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)


class BatchVideoInput(BaseModel):
    video_path: str
    prompt: str
    # Accepted for client compatibility; the rendered prompt already includes it.
    transcript: Optional[str] = None


class BatchVideosRequest(BaseModel):
    videos: List[BatchVideoInput]
    fps: float = 5.0
    max_pixels: int = 360 * 420
    max_new_tokens: int = 8192
    batch_strategy: Literal["linear", "static", "bucketed", "hybrid"] = "hybrid"
    max_batch_size: int = 4
    padding_ratio_threshold: float = 1.25


def estimate_video_batch_item(index, video_path, prompt, fps, max_pixels):
    """Estimate relative multimodal token cost for batch scheduling."""
    duration = 0.0
    source_frames = 0
    container = None
    try:
        container = av.open(video_path)
        if container.duration is not None:
            duration = float(container.duration / av.time_base)
        elif container.streams.video:
            video_stream = container.streams.video[0]
            if video_stream.duration is not None:
                duration = float(video_stream.duration * video_stream.time_base)
        if container.streams.video:
            video_stream = container.streams.video[0]
            source_frames = int(video_stream.frames or 0)
            if source_frames <= 0 and video_stream.average_rate and duration > 0:
                source_frames = max(
                    1, round(duration * float(video_stream.average_rate))
                )
    except Exception as exc:
        write_log(
            f"Could not read video duration for {video_path}: {exc}; "
            "using a one-frame scheduling estimate",
            "WARNING",
        )
    finally:
        if container is not None:
            container.close()

    estimated_frames = max(1, math.ceil(duration * fps))
    video_processor = getattr(processor, "video_processor", None)
    max_frames = getattr(video_processor, "max_frames", None)
    if isinstance(max_frames, int) and max_frames > 0:
        estimated_frames = min(estimated_frames, max_frames)

    patch_size = int(getattr(video_processor, "patch_size", 16))
    merge_size = int(getattr(video_processor, "merge_size", 2))
    temporal_patch_size = int(
        getattr(video_processor, "temporal_patch_size", 2)
    )
    minimum_frames = int(getattr(video_processor, "min_frames", 4))
    requires_minimum_frames = 0 < source_frames < temporal_patch_size
    if requires_minimum_frames:
        estimated_frames = max(minimum_frames, temporal_patch_size)
    pixels_per_visual_token = (
        patch_size * patch_size * merge_size * merge_size * temporal_patch_size
    )
    visual_tokens_per_frame = max(
        1, math.ceil(max_pixels / pixels_per_visual_token)
    )
    prompt_tokens = len(
        processor.tokenizer.encode(prompt, add_special_tokens=False)
    )
    estimated_cost = estimated_frames * visual_tokens_per_frame + prompt_tokens

    return {
        "index": index,
        "video_path": video_path,
        "prompt": prompt,
        "duration": duration,
        "source_frames": source_frames,
        "estimated_frames": estimated_frames,
        "minimum_frames": minimum_frames,
        "requires_minimum_frames": requires_minimum_frames,
        "visual_tokens_per_frame": visual_tokens_per_frame,
        "prompt_tokens": prompt_tokens,
        "estimated_cost": estimated_cost,
    }


def batch_padding_ratio(items):
    """Return padded cost divided by useful cost for a proposed batch."""
    if not items:
        return 1.0
    costs = [max(1, item["estimated_cost"]) for item in items]
    return len(costs) * max(costs) / sum(costs)


def schedule_video_batches(items, strategy, max_batch_size, padding_threshold):
    """Create linear, static, bucketed, or padding-aware hybrid batches."""
    if strategy == "linear":
        return [[item] for item in items]

    minimum_frame_items = [
        item for item in items if item.get("requires_minimum_frames")
    ]
    scheduled_items = [
        item for item in items if not item.get("requires_minimum_frames")
    ]
    if strategy in {"bucketed", "hybrid"}:
        scheduled_items.sort(key=lambda item: item["estimated_cost"])

    batches = [
        scheduled_items[start : start + max_batch_size]
        for start in range(0, len(scheduled_items), max_batch_size)
    ]
    if strategy != "hybrid":
        return batches + [[item] for item in minimum_frame_items]

    def split_if_needed(batch):
        if len(batch) <= 1 or batch_padding_ratio(batch) <= padding_threshold:
            return [batch]
        midpoint = len(batch) // 2
        return split_if_needed(batch[:midpoint]) + split_if_needed(batch[midpoint:])

    hybrid_batches = []
    for batch in batches:
        hybrid_batches.extend(split_if_needed(batch))
    return hybrid_batches + [[item] for item in minimum_frame_items]


@app.post("/infer_batch_videos_path")
async def infer_batch_videos_path(request: BatchVideosRequest):
    try:
        write_log(
            f"Received batch videos request: videos={len(request.videos)}, "
            f"strategy={request.batch_strategy}, "
            f"max_batch_size={request.max_batch_size}, fps={request.fps}, "
            f"max_pixels={request.max_pixels}, "
            f"max_new_tokens={request.max_new_tokens}"
        )
        write_debug_log(f"Request parameters: {request.dict()}")

        if model is None or processor is None:
            load_models()

        if not request.videos:
            raise HTTPException(status_code=400, detail="videos cannot be empty.")
        if request.max_batch_size < 1:
            raise HTTPException(
                status_code=400, detail="max_batch_size must be at least 1."
            )
        if request.padding_ratio_threshold < 1.0:
            raise HTTPException(
                status_code=400,
                detail="padding_ratio_threshold must be at least 1.0.",
            )
        if request.fps <= 0:
            raise HTTPException(status_code=400, detail="fps must be greater than 0.")

        def strip_file_uri(path: str) -> str:
            return path[7:] if path.startswith("file://") else path

        def normalize_path(path: str) -> str:
            path = strip_file_uri(path)
            path = path.replace("/home/broliang", "/data/shared/Qwen")
            path = path.replace("~/", "/data/shared/Qwen/")
            return re.sub(
                r"^/mnt/omv/resources/video_summarizer(?=/|$)(.*)",
                r"/data/shared/Qwen/videos/video_summarizer\1",
                path,
            )

        scheduled_items = []
        for index, video in enumerate(request.videos):
            video_path = normalize_path(video.video_path)
            if not os.path.exists(video_path):
                message = f"Video not found: {video_path}"
                write_log(message, "ERROR")
                raise HTTPException(status_code=400, detail=message)
            if not video.prompt.strip():
                message = f"Prompt cannot be empty for video: {video_path}"
                write_log(message, "ERROR")
                raise HTTPException(status_code=400, detail=message)
            scheduled_items.append(
                estimate_video_batch_item(
                    index,
                    video_path,
                    video.prompt,
                    request.fps,
                    request.max_pixels,
                )
            )

        batches = schedule_video_batches(
            scheduled_items,
            request.batch_strategy,
            request.max_batch_size,
            request.padding_ratio_threshold,
        )
        write_log(
            f"Scheduled {len(scheduled_items)} videos into {len(batches)} batches: "
            + ", ".join(
                f"size={len(batch)}/padding={batch_padding_ratio(batch):.3f}"
                for batch in batches
            )
        )

        results = [None] * len(scheduled_items)
        valid_caption_count = 0
        total_preprocessing_seconds = 0.0
        total_generation_seconds = 0.0

        for batch_number, batch in enumerate(batches, start=1):
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": item["video_path"]},
                            {"type": "text", "text": item["prompt"]},
                        ],
                    }
                ]
                for item in batch
            ]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            preprocessing_started = time.perf_counter()
            video_processor_kwargs = {
                "padding": True,
                "max_pixels": request.max_pixels,
            }
            if len(batch) == 1 and batch[0]["requires_minimum_frames"]:
                video_processor_kwargs["num_frames"] = batch[0]["minimum_frames"]
                video_processor_kwargs["fps"] = None
                sampling_description = (
                    f"num_frames={batch[0]['minimum_frames']} (minimum-frame fallback)"
                )
            else:
                video_processor_kwargs["fps"] = request.fps
                sampling_description = f"fps={request.fps}"

            inputs = processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=video_processor_kwargs,
            )
            inputs = inputs.to(model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preprocessing_seconds = time.perf_counter() - preprocessing_started
            total_preprocessing_seconds += preprocessing_seconds

            attention_tokens = inputs.attention_mask.sum(dim=1).tolist()
            padded_input_shape = tuple(inputs.input_ids.shape)
            video_grid_t = (
                inputs.video_grid_thw[:, 0].tolist()
                if "video_grid_thw" in inputs
                else [None] * len(batch)
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            generation_started = time.perf_counter()
            generated_ids = model.generate(
                **inputs, max_new_tokens=request.max_new_tokens
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            total_generation_seconds += generation_seconds
            peak_cuda_gb = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            pad_token_id = processor.tokenizer.pad_token_id
            generated_token_counts = [
                int((out_ids != pad_token_id).sum().item())
                if pad_token_id is not None
                else int(out_ids.numel())
                for out_ids in generated_ids_trimmed
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            write_log(
                f"Batch {batch_number}/{len(batches)} metrics: size={len(batch)}, "
                f"padding_ratio={batch_padding_ratio(batch):.3f}, "
                f"sampling={sampling_description}, "
                f"preprocessing_seconds={preprocessing_seconds:.3f}, "
                f"generation_seconds={generation_seconds:.3f}, "
                f"padded_input_shape={padded_input_shape}, "
                f"attention_tokens={attention_tokens}, "
                f"generated_tokens={generated_token_counts}, "
                f"peak_cuda_gb={peak_cuda_gb:.3f}"
            )

            for position, item in enumerate(batch):
                caption = output_texts[position] if position < len(output_texts) else ""
                caption_length = len(caption)
                write_log(
                    f"Batch item metrics: video={item['video_path']}, "
                    f"duration={item['duration']:.3f}, "
                    f"source_frames={item['source_frames']}, "
                    f"estimated_frames={item['estimated_frames']}, "
                    f"video_grid_t={video_grid_t[position]}, "
                    f"prompt_tokens={item['prompt_tokens']}, "
                    f"estimated_cost={item['estimated_cost']}, "
                    f"attention_tokens={attention_tokens[position]}, "
                    f"generated_tokens={generated_token_counts[position]}, "
                    f"caption_length={caption_length}"
                )
                write_debug_log(
                    f"Raw batch caption: video={item['video_path']}, "
                    f"length={caption_length}, raw={caption!r}"
                )

                validation_error = None
                if not caption.strip():
                    validation_error = "empty output"
                else:
                    try:
                        payload = json.loads(caption)
                    except json.JSONDecodeError as exc:
                        validation_error = f"invalid JSON: {exc}"
                    else:
                        if not isinstance(payload, dict):
                            validation_error = "JSON root is not an object"
                        elif not str(payload.get("description") or "").strip():
                            validation_error = "missing nonempty description"

                if validation_error:
                    write_log(
                        f"Invalid batch caption: video={item['video_path']}, "
                        f"length={caption_length}, reason={validation_error}",
                        "WARNING",
                    )
                else:
                    valid_caption_count += 1

                results[item["index"]] = {
                    "video": item["video_path"],
                    "prompt": item["prompt"],
                    "caption": caption,
                }

        write_log(
            f"Batch request complete: decoded={len(results)}, "
            f"valid={valid_caption_count}, "
            f"invalid={len(results) - valid_caption_count}, "
            f"preprocessing_seconds={total_preprocessing_seconds:.3f}, "
            f"generation_seconds={total_generation_seconds:.3f}"
        )
        return {"caption": results}

    except HTTPException:
        raise
    except Exception as e:
        error_msg = (
            f"Error in batch video inference: {str(e)}\n{traceback.format_exc()}"
        )
        write_log(error_msg, "ERROR")
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    write_log("Starting Qwen VL server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
