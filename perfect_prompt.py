import click

import flux
import fluxpro
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
    default="flux",
    type=click.Choice(["flux", "flux-pro"]),
    help="Model to use for generating images",
)
def generate_and_refine(
    prompt_path, iterations, comfy_output_dir, refine_model, gen_model
):
    with open(prompt_path, "r") as file:
        initial_prompt = file.read().strip()

    current_prompt = initial_prompt
    previous_attempts = []
    for i in range(iterations):
        click.echo(f"Iteration {i+1}/{iterations}")
        click.echo(f"Prompt: {current_prompt}")

        if gen_model == "flux":
            image_module = flux
        elif gen_model == "flux-pro":
            image_module = fluxpro
        else:
            assert False, f"Unexpected gen_model value: {gen_model}"

        current_image_path = image_module.generate_image(
            current_prompt, comfy_output_dir
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
