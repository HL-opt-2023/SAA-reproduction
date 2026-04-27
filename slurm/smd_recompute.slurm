#!/bin/bash
#SBATCH --job-name=saa_smdfix
#SBATCH --account=liu.h
#SBATCH --qos=liu.h-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=2:00:00
#SBATCH --output=/orange/liu.h/hongcheng/saa/logs/smdfix_%j.out
#SBATCH --error=/orange/liu.h/hongcheng/saa/logs/smdfix_%j.err

set -euo pipefail
SAA_ROOT=/orange/liu.h/hongcheng/saa
EXP_DIR=${SAA_ROOT}/exp2_utility
mkdir -p ${SAA_ROOT}/logs
module purge
module load matlab/2025b
hostname
date
cd ${EXP_DIR}
matlab -nodisplay -nosplash -nodesktop -batch \
  "try, maxNumCompThreads(${SLURM_CPUS_PER_TASK:-2}); smd_recompute; catch err, disp(getReport(err)); exit(1); end; exit(0);"
date
echo "done"
