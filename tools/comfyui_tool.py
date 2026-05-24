"""
ComfyUI tool — submit workflows, poll for results, fetch images.

ComfyUI runs as Electron desktop app on Windows at http://127.0.0.1:8000
Data dir: D:\ComfyUI\  |  Output: D:\ComfyUI\output\
"""

import json
import os
import time
import uuid
import base64
from pathlib import Path

import requests

from tools.registry import registry

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", "/mnt/d/ComfyUI/output"))

# ---------------------------------------------------------------------------
# Default SDXL text-to-image workflow (Juggernaut XL)
# ---------------------------------------------------------------------------
def _build_txt2img_workflow(
    prompt: str,
    negative: str = "text, watermark, ugly, blurry, low quality",
    width: int = 512,
    height: int = 512,
    steps: int = 40,
    cfg: float = 7.5,
    seed: int = -1,
    checkpoint: str = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
) -> dict:
    if seed == -1:
        seed = int(uuid.uuid4().int % (2**32))
    return {
        "3": {
            "inputs": {"ckpt_name": checkpoint},
            "class_type": "CheckpointLoaderSimple",
        },
        "4": {
            "inputs": {
                "text": prompt,
                "clip": ["3", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "5": {
            "inputs": {
                "text": negative,
                "clip": ["3", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "6": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
            "class_type": "EmptyLatentImage",
        },
        "7": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
            "class_type": "KSampler",
        },
        "8": {
            "inputs": {"samples": ["7", 0], "vae": ["3", 2]},
            "class_type": "VAEDecode",
        },
        "9": {
            "inputs": {
                "filename_prefix": "hermes",
                "images": ["8", 0],
            },
            "class_type": "SaveImage",
        },
    }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _check_available() -> bool:
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _queue_prompt(workflow: dict, client_id: str) -> str:
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["prompt_id"]


def _poll_history(prompt_id: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        data = r.json()
        if prompt_id in data:
            return data[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI job {prompt_id} did not complete in {timeout}s")


def _fetch_image_b64(filename: str, subfolder: str = "", img_type: str = "output") -> str:
    params = {"filename": filename, "subfolder": subfolder, "type": img_type}
    r = requests.get(f"{COMFYUI_URL}/view", params=params, timeout=30)
    r.raise_for_status()
    return base64.b64encode(r.content).decode()


def _get_output_images(history: dict) -> list[dict]:
    images = []
    for node_id, node_data in history.get("outputs", {}).items():
        for img in node_data.get("images", []):
            images.append(img)
    return images


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------
def comfyui_generate(
    prompt: str,
    negative_prompt: str = "text, watermark, ugly, blurry, low quality",
    width: int = 512,
    height: int = 512,
    steps: int = 40,
    cfg: float = 7.5,
    seed: int = -1,
    checkpoint: str = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    save_path: str = "",
    task_id: str = None,
) -> str:
    if not _check_available():
        return json.dumps({
            "success": False,
            "error": "ComfyUI is not running. Launch it first or check http://127.0.0.1:8000",
        })
    try:
        client_id = str(uuid.uuid4())
        workflow = _build_txt2img_workflow(
            prompt=prompt,
            negative=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            checkpoint=checkpoint,
        )
        prompt_id = _queue_prompt(workflow, client_id)
        history = _poll_history(prompt_id, timeout=180)
        images = _get_output_images(history)

        if not images:
            return json.dumps({"success": False, "error": "No images in output", "prompt_id": prompt_id})

        img_info = images[0]
        img_b64 = _fetch_image_b64(img_info["filename"], img_info.get("subfolder", ""))

        # Save locally if path given
        saved_path = ""
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(img_b64))
            saved_path = str(p)
        else:
            from hermes_constants import get_hermes_home
            out_dir = Path(get_hermes_home()) / "image_gen"
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_path = str(out_dir / img_info["filename"])
            Path(saved_path).write_bytes(base64.b64decode(img_b64))

        return json.dumps({
            "success": True,
            "prompt_id": prompt_id,
            "filename": img_info["filename"],
            "saved_path": saved_path,
            "image_base64": img_b64[:100] + "...",  # truncate for readability
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def comfyui_submit_workflow(workflow_json: str, task_id: str = None) -> str:
    """Submit a raw ComfyUI workflow JSON string and return the output images."""
    if not _check_available():
        return json.dumps({"success": False, "error": "ComfyUI not running at http://127.0.0.1:8000"})
    try:
        workflow = json.loads(workflow_json)
        client_id = str(uuid.uuid4())
        prompt_id = _queue_prompt(workflow, client_id)
        history = _poll_history(prompt_id, timeout=300)
        images = _get_output_images(history)

        saved = []
        from hermes_constants import get_hermes_home
        out_dir = Path(get_hermes_home()) / "image_gen"
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_info in images:
            img_b64 = _fetch_image_b64(img_info["filename"], img_info.get("subfolder", ""))
            path = out_dir / img_info["filename"]
            path.write_bytes(base64.b64decode(img_b64))
            saved.append(str(path))

        return json.dumps({"success": True, "prompt_id": prompt_id, "saved_paths": saved})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def comfyui_status(task_id: str = None) -> str:
    """Check if ComfyUI is running and return system stats."""
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        if r.status_code == 200:
            stats = r.json()
            queue_r = requests.get(f"{COMFYUI_URL}/queue", timeout=3)
            queue = queue_r.json() if queue_r.status_code == 200 else {}
            return json.dumps({"running": True, "url": COMFYUI_URL, "stats": stats, "queue": queue})
        return json.dumps({"running": False, "url": COMFYUI_URL})
    except Exception:
        return json.dumps({"running": False, "url": COMFYUI_URL, "error": "Connection refused"})


def comfyui_launch(task_id: str = None) -> str:
    """Launch ComfyUI Electron app from WSL."""
    import subprocess
    exe = r"C:\Users\cody\AppData\Local\Programs\ComfyUI\ComfyUI.exe"
    ps = r"/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        subprocess.Popen([ps, "-Command", f'Start-Process "{exe}"'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        return json.dumps({"success": True, "message": "ComfyUI launch initiated. It may take 10-15s to fully start."})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def comfyui_close(task_id: str = None) -> str:
    """Close ComfyUI Electron app when done with a generation batch."""
    import subprocess
    ps = r"/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        subprocess.run(
            [ps, "-Command", "Stop-Process -Name ComfyUI -Force -ErrorAction SilentlyContinue"],
            capture_output=True, timeout=10
        )
        return json.dumps({"success": True, "message": "ComfyUI closed."})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
registry.register(
    name="comfyui_generate",
    toolset="image_gen",
    schema={
        "name": "comfyui_generate",
        "description": (
            "Generate an image using ComfyUI (local Stable Diffusion). "
            "Uses Juggernaut XL by default. Returns the saved file path. "
            "ComfyUI must be running — call comfyui_status first, or comfyui_launch to start it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Positive image prompt"},
                "negative_prompt": {"type": "string", "description": "Negative prompt (what to avoid)"},
                "width": {"type": "integer", "description": "Image width in pixels (default 512)"},
                "height": {"type": "integer", "description": "Image height in pixels (default 512)"},
                "steps": {"type": "integer", "description": "Sampling steps (default 40)"},
                "cfg": {"type": "number", "description": "CFG scale 1-20 (default 7.5)"},
                "seed": {"type": "integer", "description": "Seed (-1 for random)"},
                "checkpoint": {"type": "string", "description": "Checkpoint filename (default: Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors)"},
                "save_path": {"type": "string", "description": "Optional custom save path for output image"},
            },
            "required": ["prompt"],
        },
    },
    handler=lambda args, **kw: comfyui_generate(
        prompt=args["prompt"],
        negative_prompt=args.get("negative_prompt", "text, watermark, ugly, blurry, low quality"),
        width=args.get("width", 512),
        height=args.get("height", 512),
        steps=args.get("steps", 40),
        cfg=args.get("cfg", 7.5),
        seed=args.get("seed", -1),
        checkpoint=args.get("checkpoint", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"),
        save_path=args.get("save_path", ""),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="comfyui_submit_workflow",
    toolset="image_gen",
    schema={
        "name": "comfyui_submit_workflow",
        "description": "Submit a raw ComfyUI workflow JSON to the queue and wait for output images.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_json": {"type": "string", "description": "Full ComfyUI workflow as a JSON string"},
            },
            "required": ["workflow_json"],
        },
    },
    handler=lambda args, **kw: comfyui_submit_workflow(
        workflow_json=args["workflow_json"],
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="comfyui_status",
    toolset="image_gen",
    schema={
        "name": "comfyui_status",
        "description": "Check if ComfyUI is running and return system stats and queue info.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: comfyui_status(task_id=kw.get("task_id")),
)

registry.register(
    name="comfyui_launch",
    toolset="image_gen",
    schema={
        "name": "comfyui_launch",
        "description": "Launch the ComfyUI Electron desktop app on Windows from WSL. Wait ~15 seconds then call comfyui_status to confirm.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: comfyui_launch(task_id=kw.get("task_id")),
)

registry.register(
    name="comfyui_close",
    toolset="image_gen",
    schema={
        "name": "comfyui_close",
        "description": "Close ComfyUI Electron app when done with a generation batch to free GPU/RAM.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: comfyui_close(task_id=kw.get("task_id")),
)
