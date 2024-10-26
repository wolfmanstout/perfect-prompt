import click

import flux
import pixtral


@click.command()
@click.argument("prompt_path", type=click.Path(exists=True))
@click.option("--iterations", "-n", default=3, help="Number of refinement iterations")
@click.option(
    "--comfy-output-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory where generated images are found",
)
def generate_and_refine(prompt_path, iterations, comfy_output_dir):
    with open(prompt_path, "r") as file:
        initial_prompt = file.read().strip()

    current_prompt = initial_prompt
    previous_attempts = []
    for i in range(iterations):
        click.echo(f"Iteration {i+1}/{iterations}")
        click.echo(f"Prompt: {current_prompt}")

        current_image_path = flux.generate_image(current_prompt, comfy_output_dir)
        click.echo(f"Image: {current_image_path}")

        review, refined_prompt = pixtral.refine_prompt(
            initial_prompt, current_prompt, current_image_path, previous_attempts
        )
        click.echo(f"Review: {review}")

        previous_attempts.append((current_prompt, review))
        current_prompt = refined_prompt
        click.echo("\n\n")

    click.echo("Image generation and refinement complete.")


if __name__ == "__main__":
    generate_and_refine()
