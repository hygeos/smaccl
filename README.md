# SMACCL

SMACCL provides atmospheric correction workflows based on the SMAC model family,
with OpenCL acceleration and sensor-specific I/O utilities used in C3S/EO
processing chains.

## Documentation

Project documentation is written with Quarto and stored in `docs/`.

- Main page: `docs/index.qmd`
- Getting started: `docs/getting-started.qmd`
- Configuration reference: `docs/configuration.qmd`
- Processing workflow: `docs/workflow.qmd`
- Development and testing: `docs/development.qmd`

Render locally:

```bash
quarto render docs
```

Serve locally:

```bash
quarto preview docs
```

## Quick Start

### 1) Create a Python environment

With `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Or with `pixi`:

```bash
pixi install
```

### 2) Run processing with a config file

```bash
python -m smaccl.c3s_smaccl c3s_config.cfg
```

Other examples:

```bash
python -m smaccl.c3s_smaccl c3s_probav_config.cfg
python -m smaccl.c3s_smaccl viirs_config.cfg
```

## Tests

Run tests with:

```bash
pytest tests/
```

Or through pixi:

```bash
pixi run tests
```

## Notes

- Some bundled tests rely on external datasets/paths that may not be available on
	every machine.
- OpenCL availability and drivers are required for production-like runs.

## Reference

HYGEOS: https://hygeos.com/en/