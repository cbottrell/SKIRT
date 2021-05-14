#!/bin/bash 
#SBATCH --job-name=skirt
#SBATCH --account=rrg-jfncc_cpu
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END
#SBATCH --error='/home/bottrell/projects/def-simardl/bottrell/Simulations/IllustrisTNG/Scripts/SLURM/%x-%A_%a.err' 
#SBATCH --output='/home/bottrell/projects/def-simardl/bottrell/Simulations/IllustrisTNG/Scripts/SLURM/%x-%A_%a.out' 
#SBATCH --mail-user=connor.bottrell@ipmu.jp

source /home/bottrell/virtualenvs/tf37-cpu/bin/activate

cd /home/bottrell/projects/def-simardl/bottrell/Simulations/IllustrisTNG/Scripts/SKIRT

python run_skirt.py

