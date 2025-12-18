"""Simplified BFL Flux API client for image generation."""

import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image, PngImagePlugin

load_dotenv()

API_KEY = os.getenv("BFL_API_KEY")
BASE_URL = "https://api.bfl.ai"


def generate_image(
    prompt: str,
    output_dir: Path,
    *,
    model: str,
    width: int = 1216,
    height: int = 832,
    raw: bool = False,
    **_,
) -> Path:
    """Generate an image using the BFL Flux API.

    Args:
        prompt: Text description of the image to generate.
        output_dir: Directory where the image will be saved.
        model: Model name (e.g., flux-dev, flux-pro, flux-2-max).
        width: Output width in pixels (must be multiple of 16).
        height: Output height in pixels (must be multiple of 16).
        raw: Request raw-style image (Flux 1.x only).

    Returns:
        Path to the saved image file.
    """
    if not API_KEY:
        raise RuntimeError("BFL_API_KEY environment variable is not set")

    headers = {
        "accept": "application/json",
        "x-key": API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": 42,
        "output_format": "png",
    }

    # Flux 1.x specific parameters
    if not model.startswith("flux-2"):
        payload["prompt_upsampling"] = False
        payload["safety_tolerance"] = 6  # Max permissiveness for Flux 1.x (0-6)
        if raw:
            payload["raw"] = True
    else:
        payload["safety_tolerance"] = 5  # Max permissiveness for Flux 2.x (0-5)

    with httpx.Client(timeout=60.0) as client:
        # Submit generation request
        response = client.post(
            f"{BASE_URL}/v1/{model}",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
        polling_url = result.get(
            "polling_url", f"{BASE_URL}/v1/get_result?id={result['id']}"
        )

        # Poll for results
        while True:
            poll_response = client.get(polling_url, headers=headers)
            poll_response.raise_for_status()
            poll_data = poll_response.json()

            status = poll_data["status"]
            if status == "Ready":
                image_url = poll_data["result"]["sample"]
                image_response = client.get(image_url)
                image_response.raise_for_status()

                # Save the image with metadata
                timestamp = int(time.time() * 1000)
                output_path = output_dir / f"{model}_{timestamp}.png"
                output_path.write_bytes(image_response.content)

                image = Image.open(output_path)
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("prompt", prompt)
                metadata.add_text("model", model)
                metadata.add_text("raw", str(raw))
                image.save(output_path, "PNG", pnginfo=metadata)

                return output_path

            if status in ("Error", "Failed", "Request Moderated", "Content Moderated"):
                error_msg = poll_data.get("error", status)
                raise RuntimeError(f"Generation failed: {error_msg}")

            time.sleep(0.5)


def free_memory() -> None:
    """No-op for API version (memory is managed server-side)."""
    pass
