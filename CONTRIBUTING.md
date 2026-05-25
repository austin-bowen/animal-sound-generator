# Contributing

## Setup

After cloning the repo, create the virtual environment and install all dependencies:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Scripts

| Script                 | What it does                                                            |
|------------------------|-------------------------------------------------------------------------|
| `bin/format`           | Auto-fixes lint issues and formats all code with ruff                   |
| `bin/check`            | Runs lint check, format check, and all tests; reports overall pass/fail |
| `bin/test`             | Runs unit and integration tests with coverage                           |
| `bin/unit-test`        | Runs only unit tests with coverage                                      |
| `bin/integration-test` | Runs only integration tests with coverage                               |
| `bin/build`            | Builds the distribution package (`uv build`)                            |
| `bin/publish`          | Publishes the package to PyPI (`uv publish`)                            |

## Managing Dependencies

### Add a dependency

```bash
uv add [--dev] <package>
```

### Remove a dependency

```bash
uv remove <package>
```

### Update dependencies

Update all dependencies to their latest allowed versions:

```bash
uv sync --upgrade
```

Update a specific dependency:

```bash
uv sync --upgrade-package <package>
```

## Publishing

### First-time setup

Create a `~/.pypirc` file with your PyPI API token:

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-<your-token>
```

### Release steps

1. Update the version in `pyproject.toml`
2. Commit and tag the release: `git tag v<version>`
3. Build the package: `bin/build`
4. Publish to PyPI: `bin/publish`
