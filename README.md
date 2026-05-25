# template-python-lib
A template Python library.

Setup instructions:
1. Clone the repo under a new name, e.g. with the `Use this template` button in GitHub.
2. Run: `./init`

## Tech Stack

| Tool                                                                                  | Purpose                                       |
|---------------------------------------------------------------------------------------|-----------------------------------------------|
| [uv](https://docs.astral.sh/uv/)                                                      | Virtual environment and dependency management |
| [ruff](https://docs.astral.sh/ruff/)                                                  | Linting and formatting                        |
| [pytest](https://docs.pytest.org/) + [pytest-cov](https://pytest-cov.readthedocs.io/) | Testing with coverage                         |
| [GitHub Actions](https://docs.github.com/en/actions)                                  | CI across multiple Python versions            |
| [Dependabot](https://docs.github.com/en/code-security/dependabot)                     | Automated GitHub Actions version updates      |

## Features

After running `./init`, the following will be in place:

### Project structure
- `.venv/` — created by `uv` with the lowest supported Python version specified
- `src/<lib_name>/` — library source code, created by `uv init --lib`
- `tests/unit/` — unit tests
- `tests/integration/` — integration tests

### `bin/` scripts
| Script                 | What it does                                                            |
|------------------------|-------------------------------------------------------------------------|
| `bin/format`           | Auto-fixes lint issues and formats all code with ruff                   |
| `bin/check`            | Runs lint check, format check, and all tests; reports overall pass/fail |
| `bin/test`             | Runs unit and integration tests with coverage                           |
| `bin/unit-test`        | Runs only unit tests with coverage                                      |
| `bin/integration-test` | Runs only integration tests with coverage                               |
| `bin/build`            | Builds the distribution package (`uv build`)                            |
| `bin/publish`          | Publishes the package to PyPI (`uv publish`)                            |

### Git pre-commit hook
On every commit, the hook automatically:
1. Runs `bin/format` and re-stages any changed files
2. Runs `bin/check` (proceeds even if checks fail, with a warning)

### CI (GitHub Actions)
- Runs `bin/check` on every push and pull request
- Tests against all Python versions specified during `./init`
- Uses a matrix strategy so all versions run in parallel

### Dependabot
- Checks for GitHub Actions version updates monthly