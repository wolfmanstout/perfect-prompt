import json
import os
import time
from pathlib import Path
from urllib import request

WORKFLOW_TEXT = """
{
  "6": {
    "inputs": {
      "text": "",
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp16.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-dev.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "13": {
    "inputs": {
      "noise": [
        "25",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "17",
        0
      ],
      "latent_image": [
        "27",
        0
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "simple",
      "steps": 20,
      "denoise": 1,
      "model": [
        "30",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "30",
        0
      ],
      "conditioning": [
        "26",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "25": {
    "inputs": {
      "noise_seed": 743783895749829
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "26": {
    "inputs": {
      "guidance": 3.5,
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "27": {
    "inputs": {
      "width": 1216,
      "height": 832,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage",
    "_meta": {
      "title": "EmptySD3LatentImage"
    }
  },
  "30": {
    "inputs": {
      "max_shift": 1.15,
      "base_shift": 0.5,
      "width": 1216,
      "height": 832,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "ModelSamplingFlux",
    "_meta": {
      "title": "ModelSamplingFlux"
    }
  }
}
"""


def generate_image(
    prompt,
    output_dir: Path,
    *,
    comfy_output_dir: Path,
    **_,
):
    # Get the initial list of files
    initial_files = set(comfy_output_dir.glob("*.png"))

    # Generate image with Flux
    queue_prompt(prompt)

    # Wait for the filesystem to update with the new image
    while True:
        current_files = set(comfy_output_dir.glob("*.png"))
        new_files = current_files - initial_files
        if new_files:
            latest_image = max(new_files, key=os.path.getctime)
            break
        time.sleep(5)

    return move_image_to_output(latest_image, output_dir)


def move_image_to_output(image_path: Path, output_dir: Path):
    timestamp = int(time.time() * 1000)
    output_path = output_dir / f"local-flux_{timestamp}.png"
    image_path.rename(output_path)
    return output_path


def queue_prompt(prompt):
    workflow = json.loads(WORKFLOW_TEXT)
    workflow["6"]["inputs"]["text"] = prompt
    p = {"prompt": workflow}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


def free_memory():
    p = {"unload_models": True, "free_memory": True}
    data = json.dumps(p).encode("utf-8")
    req = request.Request("http://127.0.0.1:8188/free", data=data)
    request.urlopen(req)
