from pathlib import Path

import click

from . import flux, fluxapi, refine


@click.command()
@click.version_option()
@click.argument("prompt", required=False)
@click.option(
    "--prompt-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the prompt file",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(writable=True, path_type=Path),
    help="Directory where final images will be saved",
)
@click.option("--iterations", "-n", default=3, help="Number of refinement iterations")
@click.option(
    "--refine-model",
    default="local-pixtral",
    help="Model to use for refining prompts",
)
@click.option(
    "--gen-model",
    default="local-flux",
    type=click.Choice(
        ["local-flux", "flux-pro-1.1-ultra", "flux-pro-1.1", "flux-pro", "flux-dev"]
    ),
    help="Model to use for generating images",
)
@click.option(
    "--comfy-output-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory where generated images are found (required if using local-flux)",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Request raw-style image from the Flux API",
)
@click.option(
    "--review-temperature",
    default=None,
    type=float,
    help="Temperature setting for the review prompt",
)
@click.option(
    "--refine-temperature",
    default=None,
    type=float,
    help="Temperature setting for the refine prompt",
)
def cli(
    prompt: str,
    prompt_path: Path,
    output_dir: Path,
    iterations,
    refine_model,
    gen_model,
    comfy_output_dir: Path,
    raw,
    review_temperature,
    refine_temperature,
):
    if prompt_path and prompt:
        raise click.UsageError("Cannot use both --prompt-path and --prompt options.")
    if not prompt_path and not prompt:
        raise click.UsageError("One of --prompt-path or --prompt must be set.")

    if gen_model == "local-flux" and not comfy_output_dir:
        raise click.UsageError("--comfy-output-dir is required when using local-flux.")

    initial_prompt = prompt_path.read_text().strip() if prompt_path else prompt.strip()

    output_dir.mkdir(exist_ok=True, parents=True)

    current_prompt = initial_prompt
    previous_attempts = []
    for i in range(iterations):
        click.echo(f"Iteration {i+1}/{iterations}")
        click.echo(f"Prompt: {current_prompt}")

        image_module = flux if gen_model == "local-flux" else fluxapi

        current_image_path = image_module.generate_image(
            current_prompt,
            output_dir,
            comfy_output_dir=comfy_output_dir,
            model=gen_model,
            raw=raw,
        )
        if refine_model.startswith("local"):
            # Free up memory for the local model to use
            image_module.free_memory()
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
