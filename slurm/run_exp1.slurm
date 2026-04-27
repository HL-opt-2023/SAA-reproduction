#!/bin/bash
#SBATCH --job-name=saa_exp1
#SBATCH --account=liu.h
#SBATCH --qos=liu.h-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --output=/orange/liu.h/hongcheng/saa/logs/exp1_%j.out
#SBATCH --error=/orange/liu.h/hongcheng/saa/logs/exp1_%j.err

set -euo pipefail

# ---- paths ---------------------------------------------------------
SAA_ROOT=/orange/liu.h/hongcheng/saa
EXP_DIR=${SAA_ROOT}/exp1_linreg
mkdir -p ${SAA_ROOT}/logs ${EXP_DIR}/results

# ---- modules -------------------------------------------------------
module purge
module load matlab/2025b

# ---- sanity --------------------------------------------------------
hostname
date
echo "running on $(nproc) cpus, $(free -h | awk '/Mem:/{print $2}') ram"

# ---- run -----------------------------------------------------------
cd ${EXP_DIR}

matlab -nodisplay -nosplash -nodesktop -batch \
  "try, maxNumCompThreads(${SLURM_CPUS_PER_TASK:-8}); set(0,'DefaultFigureVisible','off'); run_exp1; catch err, disp(getReport(err)); exit(1); end; exit(0);"

date
echo "done"
