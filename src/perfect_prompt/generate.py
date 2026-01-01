import copy
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib import request

import httpx
from dotenv import load_dotenv
from PIL import Image, PngImagePlugin

load_dotenv()

# Flux.1 Dev workflow
WORKFLOW_FLUX = {
    "6": {
        "inputs": {"text": "", "clip": ["11", 0]},
        "class_type": "CLIPTextEncode",
    },
    "8": {
        "inputs": {"samples": ["13", 0], "vae": ["10", 0]},
        "class_type": "VAEDecode",
    },
    "9": {
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
        "class_type": "SaveImage",
    },
    "10": {
        "inputs": {"vae_name": "ae.safetensors"},
        "class_type": "VAELoader",
    },
    "11": {
        "inputs": {
            "clip_name1": "t5xxl_fp16.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux",
        },
        "class_type": "DualCLIPLoader",
    },
    "12": {
        "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"},
        "class_type": "UNETLoader",
    },
    "13": {
        "inputs": {
            "noise": ["25", 0],
            "guider": ["22", 0],
            "sampler": ["16", 0],
            "sigmas": ["17", 0],
            "latent_image": ["27", 0],
        },
        "class_type": "SamplerCustomAdvanced",
    },
    "16": {
        "inputs": {"sampler_name": "euler"},
        "class_type": "KSamplerSelect",
    },
    "17": {
        "inputs": {
            "scheduler": "simple",
            "steps": 20,
            "denoise": 1,
            "model": ["30", 0],
        },
        "class_type": "BasicScheduler",
    },
    "22": {
        "inputs": {"model": ["30", 0], "conditioning": ["26", 0]},
        "class_type": "BasicGuider",
    },
    "25": {
        "inputs": {"noise_seed": 743783895749829},
        "class_type": "RandomNoise",
    },
    "26": {
        "inputs": {"guidance": 3.5, "conditioning": ["6", 0]},
        "class_type": "FluxGuidance",
    },
    "27": {
        "inputs": {"width": 1216, "height": 832, "batch_size": 1},
        "class_type": "EmptySD3LatentImage",
    },
    "30": {
        "inputs": {
            "max_shift": 1.15,
            "base_shift": 0.5,
            "width": 1216,
            "height": 832,
            "model": ["12", 0],
        },
        "class_type": "ModelSamplingFlux",
    },
}

# Flux.1 Krea Dev workflow
WORKFLOW_KREA = {
    "6": {
        "inputs": {"text": "", "clip": ["40", 0]},
        "class_type": "CLIPTextEncode",
    },
    "8": {
        "inputs": {"samples": ["31", 0], "vae": ["39", 0]},
        "class_type": "VAEDecode",
    },
    "9": {
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
        "class_type": "SaveImage",
    },
    "27": {
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        "class_type": "EmptySD3LatentImage",
    },
    "31": {
        "inputs": {
            "seed": 1073551260370905,
            "steps": 20,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["38", 0],
            "positive": ["6", 0],
            "negative": ["42", 0],
            "latent_image": ["27", 0],
        },
        "class_type": "KSampler",
    },
    "38": {
        "inputs": {
            "unet_name": "flux1-krea-dev.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
    },
    "39": {
        "inputs": {"vae_name": "ae.safetensors"},
        "class_type": "VAELoader",
    },
    "40": {
        "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5xxl_fp16.safetensors",
            "type": "flux",
        },
        "class_type": "DualCLIPLoader",
    },
    "42": {
        "inputs": {"conditioning": ["6", 0]},
        "class_type": "ConditioningZeroOut",
    },
}

# Z-Image-Turbo workflow
WORKFLOW_Z_IMAGE_TURBO = {
    "9": {
        "inputs": {"filename_prefix": "z-image", "images": ["43", 0]},
        "class_type": "SaveImage",
    },
    "39": {
        "inputs": {
            "clip_name": "qwen_3_4b.safetensors",
            "type": "lumina2",
            "device": "default",
        },
        "class_type": "CLIPLoader",
    },
    "40": {
        "inputs": {"vae_name": "ae.safetensors"},
        "class_type": "VAELoader",
    },
    "41": {
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        "class_type": "EmptySD3LatentImage",
    },
    "42": {
        "inputs": {"conditioning": ["45", 0]},
        "class_type": "ConditioningZeroOut",
    },
    "43": {
        "inputs": {"samples": ["44", 0], "vae": ["40", 0]},
        "class_type": "VAEDecode",
    },
    "44": {
        "inputs": {
            "seed": 917105978566282,
            "steps": 9,
            "cfg": 1,
            "sampler_name": "res_multistep",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["47", 0],
            "positive": ["45", 0],
            "negative": ["42", 0],
            "latent_image": ["41", 0],
        },
        "class_type": "KSampler",
    },
    "45": {
        "inputs": {"text": "", "clip": ["39", 0]},
        "class_type": "CLIPTextEncode",
    },
    "46": {
        "inputs": {
            "unet_name": "z_image_turbo_bf16.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
    },
    "47": {
        "inputs": {"shift": 3, "model": ["46", 0]},
        "class_type": "ModelSamplingAuraFlow",
    },
}


@dataclass
class WorkflowConfig:
    workflow: dict
    prompt_node_id: str


COMFYUI_WORKFLOWS = {
    "comfyui-flux": WorkflowConfig(workflow=WORKFLOW_FLUX, prompt_node_id="6"),
    "comfyui-flux-krea": WorkflowConfig(workflow=WORKFLOW_KREA, prompt_node_id="6"),
    "comfyui-z-image-turbo": WorkflowConfig(
        workflow=WORKFLOW_Z_IMAGE_TURBO, prompt_node_id="45"
    ),
}

BFL_MODELS = frozenset(
    {
        "flux-pro-1.1-ultra",
        "flux-pro-1.1",
        "flux-pro",
        "flux-dev",
        "flux-2-max",
        "flux-2-pro",
        "flux-2-flex",
    }
)


class ImageGenerator(ABC):
    @abstractmethod
    def generate_image(self, prompt: str, output_dir: Path, **kwargs) -> Path:
        pass

    @abstractmethod
    def free_memory(self) -> None:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class ComfyUIGenerator(ImageGenerator):
    def __init__(
        self,
        model: str,
        config: WorkflowConfig,
        comfyui_url: str = "http://127.0.0.1:8000",
    ):
        self._model = model
        self._config = config
        self._comfyui_url = comfyui_url

    @property
    def model_name(self) -> str:
        return self._model

    def generate_image(
        self,
        prompt: str,
        output_dir: Path,
        *,
        comfyui_output_dir: Path,
        **_,
    ) -> Path:
        initial_files = set(comfyui_output_dir.glob("*.png"))

        self._queue_prompt(prompt)

        while True:
            current_files = set(comfyui_output_dir.glob("*.png"))
            new_files = current_files - initial_files
            if new_files:
                latest_image = max(new_files, key=os.path.getctime)
                break
            time.sleep(5)

        return self._move_image_to_output(latest_image, output_dir)

    def _queue_prompt(self, prompt: str) -> None:
        workflow = copy.deepcopy(self._config.workflow)
        workflow[self._config.prompt_node_id]["inputs"]["text"] = prompt

        data = json.dumps({"prompt": workflow}).encode("utf-8")
        req = request.Request(f"{self._comfyui_url}/prompt", data=data)
        request.urlopen(req)

    def _move_image_to_output(self, image_path: Path, output_dir: Path) -> Path:
        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"{self._model}_{timestamp}.png"
        image_path.rename(output_path)
        return output_path

    def free_memory(self) -> None:
        data = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
        req = request.Request(f"{self._comfyui_url}/free", data=data)
        request.urlopen(req)


class BFLAPIGenerator(ImageGenerator):
    def __init__(self, model: str, api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.getenv("BFL_API_KEY")
        self._base_url = "https://api.bfl.ai"

    @property
    def model_name(self) -> str:
        return self._model

    def generate_image(
        self,
        prompt: str,
        output_dir: Path,
        *,
        width: int = 1216,
        height: int = 832,
        raw: bool = False,
        **_,
    ) -> Path:
        if not self._api_key:
            raise ValueError("BFL_API_KEY environment variable is not set")
        headers = {
            "accept": "application/json",
            "x-key": self._api_key,
            "Content-Type": "application/json",
        }

        payload: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": 42,
            "output_format": "png",
        }

        # Flux 1.x specific parameters
        if not self._model.startswith("flux-2"):
            payload["prompt_upsampling"] = False
            payload["safety_tolerance"] = 6  # Max permissiveness for Flux 1.x (0-6)
            if raw:
                payload["raw"] = True
        else:
            payload["safety_tolerance"] = 5  # Max permissiveness for Flux 2.x (0-5)

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self._base_url}/v1/{self._model}",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            polling_url = result.get(
                "polling_url", f"{self._base_url}/v1/get_result?id={result['id']}"
            )

            while True:
                poll_response = client.get(polling_url, headers=headers)
                poll_response.raise_for_status()
                poll_data = poll_response.json()

                status = poll_data["status"]
                if status == "Ready":
                    image_url = poll_data["result"]["sample"]
                    image_response = client.get(image_url)
                    image_response.raise_for_status()

                    timestamp = int(time.time() * 1000)
                    output_path = output_dir / f"{self._model}_{timestamp}.png"
                    output_path.write_bytes(image_response.content)

                    image = Image.open(output_path)
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("prompt", prompt)
                    metadata.add_text("model", self._model)
                    metadata.add_text("raw", str(raw))
                    image.save(output_path, "PNG", pnginfo=metadata)

                    return output_path

                if status in ("Error", "Failed", "Request Moderated", "Content Moderated"):
                    error_msg = poll_data.get("error", status)
                    raise RuntimeError(f"Generation failed: {error_msg}")

                time.sleep(0.5)

    def free_memory(self) -> None:
        pass


def get_generator(model: str) -> ImageGenerator:
    if model in COMFYUI_WORKFLOWS:
        config = COMFYUI_WORKFLOWS[model]
        return ComfyUIGenerator(model=model, config=config)

    if model in BFL_MODELS:
        return BFLAPIGenerator(model=model)

    available = list(COMFYUI_WORKFLOWS.keys()) + list(BFL_MODELS)
    raise ValueError(f"Unknown model: {model}. Available models: {available}")
