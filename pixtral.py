import gc
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer

mistral_models_path = Path.home().joinpath("mistral_models", "Pixtral")
mistral_models_path.mkdir(parents=True, exist_ok=True)


class PixtralModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        snapshot_download(
            repo_id="mistralai/Pixtral-12B-2409",
            allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
            local_dir=mistral_models_path,
        )
        self.tokenizer = MistralTokenizer.from_file(
            f"{mistral_models_path}/tekken.json"
        )
        self.model = Transformer.from_folder(mistral_models_path)

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def process_prompt(self, completion_request, temperature=None):
        assert self.model is not None, "Model is not loaded"
        assert self.tokenizer is not None, "Tokenizer is not loaded"
        if temperature is None:
            temperature = 0.35
        encoded = self.tokenizer.encode_chat_completion(completion_request)

        images = encoded.images
        tokens = encoded.tokens

        out_tokens, _ = generate(
            [tokens],
            self.model,
            images=[images],
            max_tokens=1024,
            temperature=temperature,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

        return self.tokenizer.decode(out_tokens[0])
