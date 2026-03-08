#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=om-em-experiment
#SBATCH -c 16
#SBATCH -G 4
#SBATCH --time=2-00:00:00 
#SBATCH --mem=64G
#SBATCH --nodelist=gnode085
#SBATCH --output=em_output_file.out
#SBATCH --error=em_error_file.err
#SBATCH --mail-user=trisanu.bhar@research.iiit.ac.in
#SBATCH --mail-type=ALL

# export CUDA_VISIBLE_DEVICES=0

cd 

source /ssd_scratch/anu/.venv/bin/activate

cd /home2/anu25/RSAI_Nilanjan_Sweep_Experiment

uv run --active om_em_strongreject_local.py --stage sweep --run-id full_run_14b_above 


echo "Finished !!"

