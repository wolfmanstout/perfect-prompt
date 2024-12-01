from perfect_prompt.refine import create_review_prompt, create_revision_prompt


def test_review_prompt():
    original_prompt = "a red cat sitting on a blue chair"
    expected = """The image provided was generated from the following prompt:
a red cat sitting on a blue chair

Evaluate how well the generated image adhered to the prompt and its overall aesthetic quality. Describe which elements of the prompt are present and missing from the image, then finally provide an overall score from 1 (worst) to 10 (best).
"""
    assert create_review_prompt(original_prompt) == expected


def test_revision_prompt():
    original_prompt = "a red cat sitting on a blue chair"
    current_prompt = "red cat on blue chair"
    current_review = "Good color accuracy, 8/10"
    previous_pairs = [("red cat on chair", "Missing blue color, 7/10")]

    expected = """We need to create a prompt for image generation that reflects the following intent:
a red cat sitting on a blue chair

Here are the previous prompt attempts, and how well each performed:
Prompt #1: red cat on chair
Prompt #1 review: Missing blue color, 7/10

Prompt #2: red cat on blue chair
Prompt #2 review: Good color accuracy, 8/10

Write a new prompt to generate an image that captures all the elements of the original intent better than any of the previous attempts. Be creative; do not repeat any existing prompt. Output only the new prompt, with no intro or surrounding quotes.
"""
    actual = create_revision_prompt(
        original_prompt, current_prompt, current_review, previous_pairs
    )
    assert actual == expected
