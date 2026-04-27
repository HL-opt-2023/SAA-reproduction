#!/bin/bash
# Sync SAA project to HiPerGator and submit both experiments.
# Usage:
#   bash deploy.sh            # sync only
#   bash deploy.sh submit     # sync and sbatch both experiments

set -euo pipefail

REMOTE=${HPG_REMOTE:-hpg}                                # SSH alias or user@hpg.rc.ufl.edu
LOCAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_ROOT=${SAA_ROOT:-/path/to/your/saa}

echo "Syncing ${LOCAL_ROOT} → ${REMOTE}:${REMOTE_ROOT}"
ssh "${REMOTE}" "mkdir -p ${REMOTE_ROOT}/logs"
rsync -avz --delete \
  --exclude='.git' --exclude='results' --exclude='cache' --exclude='logs' \
  --exclude='*.mat' --exclude='*.fig' --exclude='*.out' --exclude='*.err' \
  "${LOCAL_ROOT}/" "${REMOTE}:${REMOTE_ROOT}/"

if [[ "${1:-}" == "submit" ]]; then
    echo "Submitting SLURM jobs..."
    ssh "${REMOTE}" "cd ${REMOTE_ROOT}/slurm && sbatch run_exp1.slurm && sbatch run_exp2.slurm"
    echo
    echo "Check progress:"
    echo "  ssh ${REMOTE} 'squeue -u \$USER'"
    echo "  ssh ${REMOTE} 'tail -f ${REMOTE_ROOT}/logs/exp1_*.out'"
fi
