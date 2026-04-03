#!/bin/bash
#SBATCH --job-name=previa_job
#SBATCH --nodes=1	
#SBATCH --ntasks=1	
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=0-1:00:00
#SBATCH --gres=gpu:a100:3                 
#SBATCH --output=log_files/logs/llm_predict_%j.log	
#SBATCH --error=log_files/errors/llm_predict_%j.err 

# Setup conda
eval "$(conda shell.bash hook)"
conda activate "your_environment"

# Your actual job command(s) go here
python /path/to/src/code/previa_pipeline.py --config-dir ./demo/configs --file-end-name demo_data --outcome clinical

