#!/bin/bash
#SBATCH --job-name=saa_swe1
#SBATCH --account=liu.h
#SBATCH --qos=liu.h-b
#SBATCH --array=1-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH --output=/orange/liu.h/hongcheng/saa/logs/swe1_%A_%a.out
#SBATCH --error=/orange/liu.h/hongcheng/saa/logs/swe1_%A_%a.err

set -euo pipefail
SAA_ROOT=/orange/liu.h/hongcheng/saa
EXP_DIR=${SAA_ROOT}/exp1_linreg
mkdir -p ${SAA_ROOT}/logs ${EXP_DIR}/cache/sweep
module purge
module load matlab/2025b
hostname
date
echo "task ${SLURM_ARRAY_TASK_ID} of array ${SLURM_ARRAY_JOB_ID}"
cd ${EXP_DIR}
export SWEEP_TASK_IDX=${SLURM_ARRAY_TASK_ID}
matlab -nodisplay -nosplash -nodesktop -batch \
  "try, maxNumCompThreads(${SLURM_CPUS_PER_TASK:-1}); set(0,'DefaultFigureVisible','off'); run_exp1; catch err, disp(getReport(err)); exit(1); end; exit(0);"
date
echo "done"
