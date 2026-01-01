from pathlib import Path

import click

from . import refine
from .generate import get_generator


@click.command()
@click.version_option()
@click.argument("prompt", required=True)
@click.option(
    "--from-file",
    is_flag=True,
    help="Treat prompt argument as file path instead of literal text",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(writable=True, path_type=Path),
    help="Directory where final images will be saved",
)
@click.option(
    "--iterations",
    "-n",
    default=3,
    show_default=True,
    help="Number of refinement iterations",
)
@click.option(
    "--refine-model",
    default="ministral-3:14b",
    show_default=True,
    help="Model to use for refining prompts",
)
@click.option(
    "--gen-model",
    default="comfyui-flux",
    show_default=True,
    type=click.Choice(
        [
            "comfyui-flux",
            "comfyui-flux-krea",
            "comfyui-z-image-turbo",
            "flux-pro-1.1-ultra",
            "flux-pro-1.1",
            "flux-pro",
            "flux-dev",
            "flux-2-max",
            "flux-2-pro",
            "flux-2-flex",
        ]
    ),
    help="Model to use for generating images",
)
@click.option(
    "--comfyui-output-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory where generated images are found (required if using comfyui-* models)",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Request raw-style image from the Flux API",
)
@click.option(
    "--review-temperature",
    type=float,
    help="Temperature setting for the review prompt",
)
@click.option(
    "--refine-temperature",
    type=float,
    help="Temperature setting for the refine prompt",
)
@click.option(
    "--free-vram",
    is_flag=True,
    help="Free image generation VRAM before running refine model",
)
def cli(
    prompt: str,
    from_file: bool,
    output_dir: Path,
    iterations,
    refine_model,
    gen_model,
    comfyui_output_dir: Path,
    raw,
    review_temperature,
    refine_temperature,
    free_vram,
):
    if from_file:
        prompt = Path(prompt).read_text()

    if gen_model.startswith("comfyui-") and not comfyui_output_dir:
        raise click.UsageError(
            "--comfyui-output-dir is required when using comfyui-* models."
        )

    initial_prompt = prompt.strip()

    output_dir.mkdir(exist_ok=True, parents=True)

    current_prompt = initial_prompt
    previous_attempts = []
    for i in range(iterations):
        click.echo(f"Iteration {i + 1}/{iterations}")
        click.echo(f"Prompt: {current_prompt}")

        generator = get_generator(gen_model)

        current_image_path = generator.generate_image(
            current_prompt,
            output_dir,
            comfyui_output_dir=comfyui_output_dir,
            raw=raw,
        )
        if free_vram:
            generator.free_memory()
        click.echo(f"Image: {current_image_path}")

        review, refined_prompt = refine.refine_prompt(
            initial_prompt,
            current_prompt,
            current_image_path,
            previous_attempts,
            refine_model=refine_model,
            review_temperature=review_temperature,
            refine_temperature=refine_temperature,
        )
        click.echo(f"Review: {review}")

        previous_attempts.append((current_prompt, review))
        current_prompt = refined_prompt
        click.echo("\n\n")

    click.echo("Image generation and refinement complete.")
