import json
import os
import time
from pathlib import Path
from urllib import request

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

WORKFLOWS = {
    "comfyui-flux": WORKFLOW_FLUX,
    "comfyui-flux-krea": WORKFLOW_KREA,
}


def generate_image(
    prompt,
    output_dir: Path,
    *,
    comfyui_output_dir: Path,
    model: str = "comfyui-flux",
    **_,
):
    # Get the initial list of files
    initial_files = set(comfyui_output_dir.glob("*.png"))

    # Generate image with the specified workflow
    queue_prompt(prompt, model)

    # Wait for the filesystem to update with the new image
    while True:
        current_files = set(comfyui_output_dir.glob("*.png"))
        new_files = current_files - initial_files
        if new_files:
            latest_image = max(new_files, key=os.path.getctime)
            break
        time.sleep(5)

    return move_image_to_output(latest_image, output_dir, model)


def move_image_to_output(image_path: Path, output_dir: Path, model: str):
    timestamp = int(time.time() * 1000)
    output_path = output_dir / f"{model}_{timestamp}.png"
    image_path.rename(output_path)
    return output_path


def queue_prompt(prompt, model: str = "comfyui-flux"):
    import copy

    workflow = copy.deepcopy(WORKFLOWS[model])
    workflow["6"]["inputs"]["text"] = prompt

    p = {"prompt": workflow}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8000/prompt", data=data)
    request.urlopen(req)


def free_memory():
    p = {"unload_models": True, "free_memory": True}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8000/free", data=data)
    request.urlopen(req)
