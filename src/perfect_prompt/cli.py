import click

from . import flux, fluxapi, refine


@click.command()
@click.version_option()
@click.argument("prompt_path", type=click.Path(exists=True))
@click.option("--iterations", "-n", default=3, help="Number of refinement iterations")
@click.option(
    "--comfy-output-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory where generated images are found",
)
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
    prompt_path,
    iterations,
    comfy_output_dir,
    refine_model,
    gen_model,
    raw,
    review_temperature,
    refine_temperature,
):
    with open(prompt_path) as file:
        initial_prompt = file.read().strip()

    current_prompt = initial_prompt
    previous_attempts = []
    for i in range(iterations):
        click.echo(f"Iteration {i+1}/{iterations}")
        click.echo(f"Prompt: {current_prompt}")

        image_module = flux if gen_model == "local-flux" else fluxapi

        current_image_path = image_module.generate_image(
            current_prompt, comfy_output_dir, model=gen_model, raw=raw
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
