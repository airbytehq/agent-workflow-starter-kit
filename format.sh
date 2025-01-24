#!/bin/sh -e
set -x

poetry run ruff check app --fix
poetry run ruff format app
poetry run mypy app --explicit-package-bases