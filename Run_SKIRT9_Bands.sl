#!/bin/bash -l 
#SBATCH --job-name=run_skirt
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240GB
#SBATCH --array=100-101
#SBATCH --partition=p.large
#SBATCH --mail-type=END
#SBATCH --error='/u/bconn/Scratch/%x-%A_%a.err' 
#SBATCH --output='/u/bconn/Scratch/%x-%A_%a.out' 
#SBATCH --mail-user=connor.bottrell@ipmu.jp

module purge
export HOME=/u/bconn
export PATH=$HOME/conda-envs:$PATH
source $HOME/.bashrc
conda activate tf39_cpu

export SIM='TNG100-1'
export SNAP=91
export JOB_ARRAY_SIZE=300
export JOB_ARRAY_INDEX=$SLURM_ARRAY_TASK_ID

export SKIRT_NTASKS=4
export SKIRT_CPUS_PER_TASK=9
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /u/bconn/Projects/Simulations/IllustrisTNG/Scripts/SKIRT
python Run_SKIRT9_Bands.py 
