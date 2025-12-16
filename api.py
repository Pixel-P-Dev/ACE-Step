import asyncio
import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from acestep.pipeline_ace_step import ACEStepPipeline


# ------------------ FastAPI Setup ------------------
app = FastAPI(title="ACEStep Inference API")


# ------------------ Request Schema ------------------
class InferRequest(BaseModel):
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    manual_seeds: List[int]
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]

    guidance_scale_text: Optional[float] = 0.0
    guidance_scale_lyric: Optional[float] = 0.0
    save_path: Optional[str] = None


# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "your_model")
if not os.path.exists(CHECKPOINT_DIR):
    raise FileNotFoundError(
        f"Missing: {CHECKPOINT_DIR}\nCreate 'checkpoints/your_model' next to this file."
    )

# Static serving
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# ------------------ Load Model Once ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Loading ACEStep model from: {CHECKPOINT_DIR}")

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",
    torch_compile=False,
    cpu_offload=False,
    overlapped_decode=False,
)

print("Model ready.")


# ------------------ Concurrency Lock ------------------
model_lock = asyncio.Lock()

# Prevents model from processing multiple inference calls at once
# (ACEStep is NOT thread-safe and will break under load)


# ------------------ Inference Endpoint ------------------
@app.post("/infer")
async def infer(request: InferRequest, raw_request: Request):
    try:
        # ---------- Cancellation-safe section ----------
        # If client disconnects (spam/rapid clicking), we cancel safely
        async def check_disconnect():
            if await raw_request.is_disconnected():
                raise asyncio.CancelledError("Client disconnected")

        # Create filename
        if not request.save_path:
            filename = f"output_{uuid.uuid4().hex}.wav"
            rel_path = f"outputs/{filename}"
            full_path = os.path.join(OUTPUTS_DIR, filename)
        else:
            rel_path = request.save_path.replace("\\", "/")
            full_path = os.path.join(BASE_DIR, request.save_path)

        # ---------- Inference (exclusive lock) ----------
        async with model_lock:
            # Allows cancellation mid-run
            await check_disconnect()

            # Heavy operation moved to thread executor
            def run_model():
                return model(
                    audio_duration=request.audio_duration,
                    prompt=request.prompt,
                    lyrics=request.lyrics,
                    infer_step=request.infer_step,
                    guidance_scale=request.guidance_scale,
                    scheduler_type=request.scheduler_type,
                    cfg_type=request.cfg_type,
                    omega_scale=request.omega_scale,
                    manual_seeds=", ".join(map(str, request.manual_seeds)),
                    guidance_interval=request.guidance_interval,
                    guidance_interval_decay=request.guidance_interval_decay,
                    min_guidance_scale=request.min_guidance_scale,
                    use_erg_tag=request.use_erg_tag,
                    use_erg_lyric=request.use_erg_lyric,
                    use_erg_diffusion=request.use_erg_diffusion,
                    oss_steps=", ".join(map(str, request.oss_steps)),
                    guidance_scale_text=request.guidance_scale_text,
                    guidance_scale_lyric=request.guidance_scale_lyric,
                    save_path=full_path,
                )

            # Run without blocking event loop
            await asyncio.get_event_loop().run_in_executor(None, run_model)

        return {"status": "success", "save_path": rel_path}

    except asyncio.CancelledError:
        raise HTTPException(499, "Request cancelled by client")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
