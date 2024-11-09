import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BFL_API_KEY")
BASE_URL = "https://api.bfl.ml"
HEADERS = {"x-key": API_KEY}


def generate_image(prompt, output_dir, model, width=1216, height=832, raw=False):
    # Submit generation request
    with httpx.Client() as client:
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": 42,
            "output_format": "png",
            "prompt_upsampling": False,
            "safety_tolerance": 6,
        }
        if raw:
            payload["raw"] = True

        response = client.post(
            f"{BASE_URL}/v1/{model}",
            headers=HEADERS,
            json=payload,
        )
        response.raise_for_status()
        task_id = response.json()["id"]

        # Poll for results
        while True:
            result_response = client.get(
                f"{BASE_URL}/v1/get_result", params={"id": task_id}
            )
            result_response.raise_for_status()
            result_data = result_response.json()

            if result_data["status"] == "Ready":
                # Download the image
                image_url = result_data["result"]["sample"]
                image_response = client.get(image_url)
                image_response.raise_for_status()

                # Save the image
                timestamp = int(time.time() * 1000)
                output_path = Path(output_dir) / f"{model}_{timestamp}.png"
                output_path.write_bytes(image_response.content)
                return output_path

            elif result_data["status"] in [
                "Error",
                "Request Moderated",
                "Content Moderated",
            ]:
                raise Exception(
                    f"Generation failed with status: {result_data['status']}"
                )

            time.sleep(1)


def free_memory():
    # Note: This is not needed for the API version as memory management
    # is handled server-side
    pass
