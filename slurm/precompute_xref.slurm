#!/bin/bash
#SBATCH --job-name=saa_xref
#SBATCH --account=liu.h
#SBATCH --qos=liu.h-b
#SBATCH --array=1-13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=4:00:00
#SBATCH --output=/orange/liu.h/hongcheng/saa/logs/xref_%A_%a.out
#SBATCH --error=/orange/liu.h/hongcheng/saa/logs/xref_%A_%a.err

set -euo pipefail

SAA_ROOT=/orange/liu.h/hongcheng/saa
EXP_DIR=${SAA_ROOT}/exp2_utility
mkdir -p ${SAA_ROOT}/logs ${EXP_DIR}/cache

module purge
module load matlab/2025b

hostname
date
echo "array task ${SLURM_ARRAY_TASK_ID} computing x_ref for d_list index ${SLURM_ARRAY_TASK_ID}"

cd ${EXP_DIR}
export XREF_ONLY_D_IDX=${SLURM_ARRAY_TASK_ID}

matlab -nodisplay -nosplash -nodesktop -batch \
  "try, maxNumCompThreads(${SLURM_CPUS_PER_TASK:-2}); set(0,'DefaultFigureVisible','off'); run_exp2; catch err, disp(getReport(err)); exit(1); end; exit(0);"

date
echo "done"
