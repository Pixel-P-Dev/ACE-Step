import asyncio
import os
import uuid
from pathlib import Path
from typing import List, Optional
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from acestep.pipeline_ace_step import ACEStepPipeline
from huggingface_hub import snapshot_download

# ------------------ WSL / Windows Path Helper ------------------
def win_to_wsl(path: str) -> str:
    r"""
    Converts Windows paths to WSL paths if needed.
    Example:
    C:\Users\X -> /mnt/c/Users/X
    """
    if ":" in path:
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path.replace("\\", "/")


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


# ------------------ Base Paths ------------------
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(win_to_wsl(str(BASE_DIR)))

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_ROOT = BASE_DIR / "checkpoints"
CHECKPOINT_DIR = CHECKPOINT_ROOT / "ACE-Step-v1-3.5B"

HF_REPO_ID = "ACE-Step/ACE-Step-v1-3.5B"

# ------------------ HuggingFace Cache Fix ------------------
os.environ["HF_HOME"] = str(BASE_DIR / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(BASE_DIR / "hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(BASE_DIR / "hf_cache")

# ------------------ Ensure Checkpoints ------------------
def ensure_checkpoints():
    if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
        print(f"✔ Using existing checkpoints: {CHECKPOINT_DIR}")
        return

    print("⬇️  Downloading ACE-Step checkpoints...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,  # NTFS + WSL safe
        resume_download=True,
    )

    print("✅ Checkpoints ready.")

ensure_checkpoints()

# ------------------ Static Serving ------------------
app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs",
)

# ------------------ Load Model ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Loading ACEStep model from: {CHECKPOINT_DIR}")

model = ACEStepPipeline(
    checkpoint_dir=str(CHECKPOINT_DIR),
    dtype="bfloat16",
    torch_compile=False,
    cpu_offload=False,
    overlapped_decode=False,
)

print("🚀 Model ready.")

# ------------------ Concurrency Lock ------------------
model_lock = asyncio.Lock()


# ------------------ Debug-enabled /infer endpoint ------------------
@app.post("/infer")
async def infer(request: InferRequest, raw_request: Request):
    try:
        print("🔹 New inference request received")
        print(f"Request payload: {request.dict()}")

        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("Client disconnected before processing")

        # ------------------ Output Path Handling ------------------
        if request.save_path:
            safe_path = win_to_wsl(request.save_path)
            full_path = Path(safe_path).resolve()
            print(f"Converted save_path: {safe_path}")
            print(f"Resolved full_path: {full_path}")

            # SECURITY: force save inside BASE_DIR
            if BASE_DIR not in full_path.parents and full_path != BASE_DIR:
                raise HTTPException(400, "Invalid save_path")
        else:
            filename = f"output_{uuid.uuid4().hex}.wav"
            full_path = OUTPUTS_DIR / filename
            print(f"No save_path provided, using default: {full_path}")

        # Robust relative path for response
        try:
            rel_path = os.path.relpath(full_path, BASE_DIR).replace("\\", "/")
        except Exception as e:
            print(f"⚠ Failed to compute relative path: {e}")
            rel_path = full_path.name

        print(f"Final rel_path to return: {rel_path}")

        # ------------------ Run Model ------------------
        async with model_lock:

            def run_model():
                try:
                    print("🔹 Running ACEStep model...")
                    model(
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
                        save_path=str(full_path),
                    )
                    print("✅ Model completed successfully")
                except Exception as e:
                    print("❌ Exception inside model:")
                    traceback.print_exc()
                    raise e  # propagate to outer except

            await asyncio.get_event_loop().run_in_executor(None, run_model)

        print(f"🔹 Returning response: {rel_path}")
        return {
            "status": "success",
            "save_path": rel_path,
        }

    except asyncio.CancelledError:
        print("⚠ Request cancelled by client")
        raise HTTPException(499, "Client disconnected")

    except Exception as e:
        print("❌ Exception in /infer endpoint:")
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")
