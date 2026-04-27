#!/bin/bash
# NOTE: edit --account=YOUR_ACCOUNT and --qos=YOUR_QOS below for your cluster.
#SBATCH --job-name=saa_swe1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --array=1-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/swe1_%A_%a.out
#SBATCH --error=logs/swe1_%A_%a.err

set -euo pipefail
SAA_ROOT=${SAA_ROOT:-/path/to/your/saa}
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
