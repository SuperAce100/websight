#!/bin/bash

source .venv/bin/activate

uv run eval_tests/showdown/download_dataset.py

uv run -m eval_tests.showdown.clicks
