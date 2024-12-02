# perfect-prompt

[![PyPI](https://img.shields.io/pypi/v/perfect-prompt.svg)](https://pypi.org/project/perfect-prompt/)
[![Changelog](https://img.shields.io/github/v/release/wolfmanstout/perfect-prompt?include_prereleases&label=changelog)](https://github.com/wolfmanstout/perfect-prompt/releases)
[![Tests](https://github.com/wolfmanstout/perfect-prompt/actions/workflows/test.yml/badge.svg)](https://github.com/wolfmanstout/perfect-prompt/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/wolfmanstout/perfect-prompt/blob/master/LICENSE)

Perfect your image generation prompt with a visual LLM.

Works best to improve adherence to complex prompts.

## Installation

Install this tool using `pip` or `pipx`:

```bash
pip install perfect-prompt
```

Optional: to run image generation locally, you will also need to
[install ComfyUI with FLUX.1-dev](https://stable-diffusion-art.com/flux-comfyui/#Flux_regular_full_model).

## Usage

Basic usage:

```
perfect-prompt "a romantic couple walking along the beach holding hands and looking lovingly at each other, wearing beachwear, sunset, detailed faces, front view, soft focus, golden hour lighting, warm natural lighting." -o images -n 3 --comfy-output-dir="C:\ComfyUI_windows_portable\ComfyUI\output"
```

This will generate 3 images attempting to adhere to this prompt within `./images`.

By default, perfect-prompt uses local models. You can configure this behavior with flags. For example, here is a version of the above prompt that uses the same models via API:

```
perfect-prompt "a romantic couple walking along the beach holding hands and looking lovingly at each other, wearing beachwear, sunset, detailed faces, front view, soft focus, golden hour lighting, warm natural lighting." -o images -n 3 --refine-model=pixtral-12b --gen-model=flux-dev
```

Since this uses APIs, you'll need keys set in your environment:

```
LLM_MISTRAL_KEY=<your key from https://console.mistral.ai/>
BFL_API_KEY=<your key from https://docs.bfl.ml/>
```

Many models are available for `--refine-model`, for example:
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
