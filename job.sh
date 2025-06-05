#!/bin/bash
#SBATCH --job-name=showdown_clicks
#SBATCH --output=showdown_clicks_%j.out
#SBATCH --error=showdown_clicks_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=hai
#SBATCH --account=ingrai
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

source ~/.bashrc
source .venv/bin/activate

uv run eval_tests/showdown/download_dataset.py

uv run -m eval_tests.showdown.clicks