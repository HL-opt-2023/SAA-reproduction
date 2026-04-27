#!/bin/bash
#SBATCH --job-name=saa_exp2
#SBATCH --account=liu.h
#SBATCH --qos=liu.h-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=36:00:00
#SBATCH --output=/orange/liu.h/hongcheng/saa/logs/exp2_%j.out
#SBATCH --error=/orange/liu.h/hongcheng/saa/logs/exp2_%j.err

set -euo pipefail

SAA_ROOT=/orange/liu.h/hongcheng/saa
EXP_DIR=${SAA_ROOT}/exp2_utility
mkdir -p ${SAA_ROOT}/logs ${EXP_DIR}/results

module purge
module load matlab/2025b

hostname
date
echo "running on $(nproc) cpus, $(free -h | awk '/Mem:/{print $2}') ram"

cd ${EXP_DIR}

matlab -nodisplay -nosplash -nodesktop -batch \
  "try, maxNumCompThreads(${SLURM_CPUS_PER_TASK:-8}); set(0,'DefaultFigureVisible','off'); run_exp2; catch err, disp(getReport(err)); exit(1); end; exit(0);"

date
echo "done"
