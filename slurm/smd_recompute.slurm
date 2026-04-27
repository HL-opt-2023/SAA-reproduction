#!/bin/bash
# NOTE: edit --account=YOUR_ACCOUNT and --qos=YOUR_QOS below for your cluster.
#SBATCH --job-name=saa_smdfix
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=2:00:00
#SBATCH --output=logs/smdfix_%j.out
#SBATCH --error=logs/smdfix_%j.err

set -euo pipefail
SAA_ROOT=${SAA_ROOT:-/path/to/your/saa}
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
