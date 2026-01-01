import textwrap
from pathlib import Path

import llm


def create_review_prompt(original_prompt: str) -> str:
    template = textwrap.dedent("""\
        The image provided was generated from the following prompt:
        {original_prompt}

        Evaluate how well the generated image adhered to the prompt and its overall aesthetic quality. Describe which elements of the prompt are present and missing from the image, then finally provide an overall score from 1 (worst) to 10 (best).
        """)
    return template.format(original_prompt=original_prompt)


def create_revision_prompt(
    original_prompt: str,
    current_prompt: str,
    current_review: str,
    previous_attempt_pairs: list,
) -> str:
    previous_attempts = "\n\n".join(
        [
            f"Prompt #{i + 1}: {pair[0]}\nPrompt #{i + 1} review: {pair[1]}"
            for i, pair in enumerate(
                previous_attempt_pairs + [(current_prompt, current_review)]
            )
        ]
    )

    template = textwrap.dedent("""\
        We need to create a prompt for image generation that reflects the following intent:
        {original_prompt}

        Here are the previous prompt attempts, and how well each performed:
        {previous_attempts}

        Write a new prompt to generate an image that captures all the elements of the original intent better than any of the previous attempts. Be creative; do not repeat any existing prompt. Output only the new prompt, with no intro or surrounding quotes.
        """)
    return template.format(
        original_prompt=original_prompt,
        previous_attempts=previous_attempts,
    )


def refine_prompt(
    original_prompt,
    current_prompt,
    current_image_path: Path,
    previous_attempt_pairs,
    refine_model,
    review_temperature=None,
    refine_temperature=None,
):
    model = llm.get_model(refine_model)

    review_prompt = create_review_prompt(original_prompt)

    review = model.prompt(
        review_prompt,
        attachments=[llm.Attachment(path=str(current_image_path))],
        temperature=review_temperature,
    ).text()

    revision_prompt = create_revision_prompt(
        original_prompt, current_prompt, review, previous_attempt_pairs
    )

    max_attempts = 3
    attempts = 0
    refined_prompt = current_prompt
    while attempts < max_attempts:
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

    return review, refined_prompt
