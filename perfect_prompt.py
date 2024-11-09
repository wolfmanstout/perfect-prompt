import click

import flux
import fluxapi
import refine


@click.command()
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
    default="local-mistral",
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
def generate_and_refine(
    prompt_path, iterations, comfy_output_dir, refine_model, gen_model, raw
):
    with open(prompt_path, "r") as file:
        initial_prompt = file.read().strip()

    current_prompt = initial_prompt
    previous_attempts = []
    for i in range(iterations):
        click.echo(f"Iteration {i+1}/{iterations}")
        click.echo(f"Prompt: {current_prompt}")

        if gen_model == "local-flux":
            image_module = flux
        else:
            image_module = fluxapi

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
        )
        click.echo(f"Review: {review}")

        previous_attempts.append((current_prompt, review))
        current_prompt = refined_prompt
        click.echo("\n\n")

    click.echo("Image generation and refinement complete.")


if __name__ == "__main__":
    generate_and_refine()
