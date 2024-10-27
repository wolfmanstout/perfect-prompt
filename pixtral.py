import base64
import gc
import textwrap
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from mistral_common.protocol.instruct.messages import (
    ImageURLChunk,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer

mistral_models_path = Path.home().joinpath("mistral_models", "Pixtral")
mistral_models_path.mkdir(parents=True, exist_ok=True)


def refine_prompt(
    original_prompt, current_prompt, current_image_path, previous_attempt_pairs
):
    snapshot_download(
        repo_id="mistralai/Pixtral-12B-2409",
        allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"],
        local_dir=mistral_models_path,
    )
    tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
    model = Transformer.from_folder(mistral_models_path)

    # Read the image file and encode it in base64
    with open(current_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    url = f"data:image/png;base64,{encoded_string}"

    review_prompt = textwrap.dedent(f"""
        The image provided was generated from the following prompt:
        {original_prompt}

        Evaluate how well the generated image adhered to the prompt and its overall aesthetic quality. Describe which elements of the prompt are present and missing from the image, then finally provide an overall score from 1 (worst) to 10 (best).
        """)

    review_request = ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[ImageURLChunk(image_url=url), TextChunk(text=review_prompt)]
            )
        ]
    )

    review = process_prompt(review_request, tokenizer, model)

    previous_attempts = "\n\n".join(
        [
            f"Prompt #{i + 1}: {pair[0]}\nPrompt #{i + 1} review: {pair[1]}"
            for i, pair in enumerate(
                previous_attempt_pairs + [(current_prompt, review)]
            )
        ]
    )
    revision_prompt = textwrap.dedent(f"""
        We need to create a prompt for image generation that reflects the following intent:
        {original_prompt}

        Here are the previous prompt attempts, and how well each performed:
        {previous_attempts}

        Write a new prompt to generate an image that captures all the elements of the original intent better than any of the previous attempts. Be creative; do not repeat any existing prompt. Output only the new prompt, with no intro or surrounding quotes.
        """)

    revision_request = ChatCompletionRequest(
        messages=[UserMessage(content=[TextChunk(text=revision_prompt)])]
    )

    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        refined_prompt = process_prompt(revision_request, tokenizer, model)
        if (
            refined_prompt not in [pair[0] for pair in previous_attempt_pairs]
            and refined_prompt != current_prompt
        ):
            break

        print("Skipping duplicate prompt")
        attempts += 1

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return review, refined_prompt


def process_prompt(completion_request, tokenizer, model):
    encoded = tokenizer.encode_chat_completion(completion_request)

    images = encoded.images
    tokens = encoded.tokens

    out_tokens, _ = generate(
        [tokens],
        model,
        images=[images],
        max_tokens=1024,
        temperature=0.35,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )

    return tokenizer.decode(out_tokens[0])
