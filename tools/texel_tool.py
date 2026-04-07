"""
Texel Studio tool — AI-powered pixel-by-pixel sprite painter.

Agent-based: draws pixels one at a time like a human artist.
Exact palette control, clean edges, game-ready output.

Runs on-demand at http://localhost:8500
Start with texel_start() before calling other tools.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import requests

from tools.registry import registry

TEXEL_URL = os.getenv("TEXEL_URL", "http://localhost:8500")
TEXEL_DIR = "/mnt/c/Users/cody/texel-studio"


def _is_running() -> bool:
    try:
        r = requests.get(f"{TEXEL_URL}/api/palettes", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def texel_start(task_id: str = None) -> str:
    """Start Texel Studio server if not already running."""
    if _is_running():
        return json.dumps({"success": True, "message": "Texel Studio already running.", "url": TEXEL_URL})
    try:
        subprocess.Popen(
            [f"{TEXEL_DIR}/venv/bin/python3", "server.py"],
            cwd=TEXEL_DIR,
            stdout=open("/tmp/texel.log", "w"),
            stderr=subprocess.STDOUT,
        )
        for _ in range(10):
            time.sleep(1.5)
            if _is_running():
                return json.dumps({"success": True, "message": "Texel Studio started.", "url": TEXEL_URL})
        return json.dumps({"success": False, "error": "Texel Studio didn't respond after 15s. Check /tmp/texel.log"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def texel_stop(task_id: str = None) -> str:
    """Stop Texel Studio server."""
    try:
        result = subprocess.run(
            ["pkill", "-f", "texel-studio/venv/bin/python3"],
            capture_output=True, text=True
        )
        return json.dumps({"success": True, "message": "Texel Studio stopped."})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def texel_status(task_id: str = None) -> str:
    """Check if Texel Studio is running."""
    running = _is_running()
    return json.dumps({"running": running, "url": TEXEL_URL if running else None})


def texel_generate(
    prompt: str,
    sprite_type: str = "block",
    width: int = 16,
    height: int = 16,
    palette_id: int = 1,
    model: str = "",
    auto_start: bool = True,
    task_id: str = None,
) -> str:
    """
    Generate a pixel art sprite using Texel Studio's AI painting agent.
    The agent places pixels one at a time for clean, game-ready output.

    sprite_type: 'block' (tileable) or 'item' (transparent background)
    Returns generation ID — use texel_get_result to poll for the finished image.
    """
    if auto_start and not _is_running():
        start_result = json.loads(texel_start())
        if not start_result.get("success"):
            return json.dumps(start_result)

    try:
        # Fetch palette colors (new API requires colors array, not palette_id)
        colors = []
        try:
            pr = requests.get(f"{TEXEL_URL}/api/palettes", timeout=5)
            if pr.status_code == 200:
                palettes = pr.json()
                for p in palettes:
                    if p.get("id") == palette_id:
                        colors = p.get("colors", [])
                        break
                if not colors and palettes:
                    colors = palettes[0].get("colors", [])
        except Exception:
            pass

        # size is the max of width/height (API uses square canvas)
        size = max(width, height)

        payload = {
            "prompt": prompt,
            "sprite_type": sprite_type,
            "size": size,
            "colors": colors,
        }
        if model:
            payload["model"] = model

        r = requests.post(f"{TEXEL_URL}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        gen_id = data.get("id") or data.get("generation_id")
        return json.dumps({
            "success": True,
            "generation_id": gen_id,
            "message": f"Generation started. Poll with texel_get_result(generation_id='{gen_id}')",
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def texel_get_result(generation_id: str, save_path: str = "", task_id: str = None) -> str:
    """Poll for a Texel Studio generation result. Returns status and image when done."""
    try:
        r = requests.get(f"{TEXEL_URL}/api/generations/{generation_id}", timeout=30)
        r.raise_for_status()
        data = r.json()

        status = data.get("status", "unknown")
        if status not in ("done", "complete", "completed"):
            return json.dumps({"success": True, "status": status, "generation_id": generation_id})

        # Get image URL
        image_url = data.get("image_url") or data.get("output_url")
        saved_path = ""
        if image_url:
            img_r = requests.get(f"{TEXEL_URL}{image_url}" if image_url.startswith("/") else image_url, timeout=30)
            if save_path:
                p = Path(save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(img_r.content)
                saved_path = str(p)
            else:
                from hermes_constants import get_hermes_home
                out_dir = Path(get_hermes_home()) / "image_gen"
                out_dir.mkdir(parents=True, exist_ok=True)
                saved_path = str(out_dir / f"texel_{generation_id}.png")
                Path(saved_path).write_bytes(img_r.content)

        return json.dumps({"success": True, "status": status, "generation_id": generation_id, "saved_path": saved_path})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def texel_list_palettes(task_id: str = None) -> str:
    """List available color palettes in Texel Studio."""
    if not _is_running():
        return json.dumps({"error": "Texel Studio not running. Call texel_start first."})
    try:
        r = requests.get(f"{TEXEL_URL}/api/palettes", timeout=5)
        r.raise_for_status()
        palettes = [{"id": p["id"], "name": p["name"], "colors": len(p.get("colors", []))} for p in r.json()]
        return json.dumps({"palettes": palettes})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
_TOOLSET = "image_gen"

registry.register(
    name="texel_start",
    toolset=_TOOLSET,
    schema={
        "name": "texel_start",
        "description": "Start the Texel Studio pixel art server (runs on localhost:8500). Call this before using other texel_ tools.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: texel_start(task_id=kw.get("task_id")),
)

registry.register(
    name="texel_stop",
    toolset=_TOOLSET,
    schema={
        "name": "texel_stop",
        "description": "Stop the Texel Studio server when no longer needed.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: texel_stop(task_id=kw.get("task_id")),
)

registry.register(
    name="texel_status",
    toolset=_TOOLSET,
    schema={
        "name": "texel_status",
        "description": "Check if Texel Studio is running.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: texel_status(task_id=kw.get("task_id")),
)

registry.register(
    name="texel_generate",
    toolset=_TOOLSET,
    schema={
        "name": "texel_generate",
        "description": (
            "Generate pixel art using Texel Studio's AI painting agent — places pixels one at a time "
            "for exact palette control and clean game-ready edges. Better than diffusion for sprites. "
            "Auto-starts server if not running. Returns a generation_id — poll with texel_get_result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to draw (e.g. 'mossy cobblestone block', 'red potion bottle')"},
                "sprite_type": {"type": "string", "enum": ["block", "item"], "description": "'block' for tileable terrain, 'item' for transparent-bg objects"},
                "width": {"type": "integer", "description": "Canvas width in pixels (default 16)"},
                "height": {"type": "integer", "description": "Canvas height in pixels (default 16)"},
                "palette_id": {"type": "integer", "description": "Palette ID from texel_list_palettes (default 1)"},
            },
            "required": ["prompt"],
        },
    },
    handler=lambda args, **kw: texel_generate(
        prompt=args["prompt"],
        sprite_type=args.get("sprite_type", "block"),
        width=args.get("width", 16),
        height=args.get("height", 16),
        palette_id=args.get("palette_id", 1),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="texel_get_result",
    toolset=_TOOLSET,
    schema={
        "name": "texel_get_result",
        "description": "Poll for a Texel Studio generation result. Returns status and saved image path when done.",
        "parameters": {
            "type": "object",
            "properties": {
                "generation_id": {"type": "string", "description": "ID returned by texel_generate"},
                "save_path": {"type": "string", "description": "Optional custom path to save the output PNG"},
            },
            "required": ["generation_id"],
        },
    },
    handler=lambda args, **kw: texel_get_result(
        generation_id=args["generation_id"],
        save_path=args.get("save_path", ""),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="texel_list_palettes",
    toolset=_TOOLSET,
    schema={
        "name": "texel_list_palettes",
        "description": "List available color palettes in Texel Studio.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    handler=lambda args, **kw: texel_list_palettes(task_id=kw.get("task_id")),
)
