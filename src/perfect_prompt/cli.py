from pathlib import Path

import click

from . import flux, fluxapi, refine


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
    default="local-pixtral",
    show_default=True,
    help="Model to use for refining prompts",
)
@click.option(
    "--gen-model",
    default="local-flux",
    show_default=True,
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
    type=float,
    help="Temperature setting for the review prompt",
)
@click.option(
    "--refine-temperature",
    type=float,
    help="Temperature setting for the refine prompt",
)
def cli(
    prompt: str,
    from_file: bool,
    output_dir: Path,
    iterations,
    refine_model,
    gen_model,
    comfy_output_dir: Path,
    raw,
    review_temperature,
    refine_temperature,
):
    if from_file:
        prompt = Path(prompt).read_text()

    if gen_model == "local-flux" and not comfy_output_dir:
        raise click.UsageError("--comfy-output-dir is required when using local-flux.")

    initial_prompt = prompt.strip()

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
