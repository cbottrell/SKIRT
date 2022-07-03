#!/bin/bash -l 
#SBATCH --job-name=run_skirt
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=240GB
#SBATCH --array=0-1
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

export SIM='TNG50-1'
export SNAP=13
export JOB_ARRAY_SIZE=2
export JOB_ARRAY_INDEX=$SLURM_ARRAY_TASK_ID

export SKIRT_NTASKS=$SLURM_NTASKS
export SKIRT_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /u/bconn/Projects/Simulations/IllustrisTNG/Scripts/SKIRT
python Run_SKIRT9_JWST.py 

