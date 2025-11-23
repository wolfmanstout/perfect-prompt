# perfect-prompt

[![PyPI](https://img.shields.io/pypi/v/perfect-prompt.svg)](https://pypi.org/project/perfect-prompt/)
[![Changelog](https://img.shields.io/github/v/release/wolfmanstout/perfect-prompt?include_prereleases&label=changelog)](https://github.com/wolfmanstout/perfect-prompt/releases)
[![Tests](https://github.com/wolfmanstout/perfect-prompt/actions/workflows/test.yml/badge.svg)](https://github.com/wolfmanstout/perfect-prompt/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/wolfmanstout/perfect-prompt/blob/master/LICENSE)

Perfect your image generation prompt with a visual LLM.

Works best to improve adherence to complex prompts.

## Installation

Install this tool using `pip`, `pipx`, or `uv`:

```bash
pip install perfect-prompt
```

Or using `uv`:

```bash
uv tool install perfect-prompt
```

Optional: to run image generation locally, you will also need to
[install ComfyUI with FLUX.1-dev](https://stable-diffusion-art.com/flux-comfyui/#Flux_regular_full_model).

## Usage

Basic usage:

```
perfect-prompt "A robot holding a bouquet of sunflowers, standing in front of a crumbling brick wall covered in graffiti." -o images -n 3 --comfy-output-dir="C:\ComfyUI_windows_portable\ComfyUI\output"
```

This will generate 3 images attempting to adhere to this prompt within `./images`.

By default, perfect-prompt uses local models. You can configure this behavior with flags. For example, here is a version of the above prompt that uses models via API:

```
perfect-prompt "A robot holding a bouquet of sunflowers, standing in front of a crumbling brick wall covered in graffiti." -o images -n 3 --refine-model=pixtral-large --gen-model=flux-dev
```

And here is the resulting sequence of images:

![Robot against a brick wall without graffiti, holding three folowers](demo/flux-dev_1736053101055.png)

![Robot against a painted brick wall, holding a bouquet of flowers](demo/flux-dev_1736053124571.png)

![Robot against a brick wall clearly painted with graffiti, holding a bouquet of flowers](demo/flux-dev_1736053150788.png)

Since this uses APIs, you'll to set need keys in your environment:

```
LLM_MISTRAL_KEY=<your key from https://console.mistral.ai/>
BFL_API_KEY=<your key from https://docs.bfl.ml/>
```

Many models are available for `--refine-model` via Simon Willison’s [`llm` package](https://github.com/simonw/llm), for example:
```
gpt-4o (uses OPENAI_API_KEY)
gpt-4o-mini (uses OPENAI_API_KEY)
pixtral-12b (uses LLM_MISTRAL_KEY)
pixtral-large (uses LLM_MISTRAL_KEY)
gemini-1.5-pro-latest (uses LLM_GEMINI_KEY)
gemini-1.5-flash-latest (uses LLM_GEMINI_KEY)
```

For help, run:

```bash
perfect-prompt --help
```

You can also use:

```bash
python -m perfect_prompt --help
```

## Development

To contribute to this tool, use [uv](https://docs.astral.sh/uv/). The following
command will establish the venv and run tests:

```bash
uv run pytest
```

To run perfect-prompt locally, use:

```bash
uv run perfect-prompt
```
