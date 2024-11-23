import base64
import textwrap

import llm
from mistral_common.protocol.instruct.messages import (
    ImageURLChunk,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from pixtral import PixtralModel


def refine_prompt(
    original_prompt,
    current_prompt,
    current_image_path,
    previous_attempt_pairs,
    refine_model,
    review_temperature=None,
    refine_temperature=None,
):
    if refine_model == "local-mistral":
        pixtral_model = PixtralModel()
        pixtral_model.load()
    else:
        model = llm.get_model(refine_model)

    review_prompt = textwrap.dedent(f"""
        The image provided was generated from the following prompt:
        {original_prompt}

        Evaluate how well the generated image adhered to the prompt and its overall aesthetic quality. Describe which elements of the prompt are present and missing from the image, then finally provide an overall score from 1 (worst) to 10 (best).
        """)

    if refine_model == "local-mistral":
        # Read the image file and encode it in base64
        with open(current_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        url = f"data:image/png;base64,{encoded_string}"
        review_request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        ImageURLChunk(image_url=url),
                        TextChunk(text=review_prompt),
                    ]
                )
            ]
        )
        review = pixtral_model.process_prompt(
            review_request, temperature=review_temperature
        )
    else:
        review = model.prompt(
            review_prompt,
            attachments=[llm.Attachment(path=current_image_path)],
            temperature=review_temperature,
        ).text()

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

    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        if refine_model == "local-mistral":
            revision_request = ChatCompletionRequest(
                messages=[UserMessage(content=[TextChunk(text=revision_prompt)])]
            )
            refined_prompt = pixtral_model.process_prompt(
                revision_request, temperature=refine_temperature
            )
        else:
            refined_prompt = model.prompt(
                revision_prompt, temperature=refine_temperature
            ).text()

        if (
            refined_prompt not in [pair[0] for pair in previous_attempt_pairs]
            and refined_prompt != current_prompt
        ):
            break

        print("Skipping duplicate prompt")
        attempts += 1

    if refine_model == "local-mistral":
        pixtral_model.unload()

    return review, refined_prompt
