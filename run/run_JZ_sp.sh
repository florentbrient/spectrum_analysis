#!/bin/bash
#SBATCH -J SPEC
#SBATCH -N 1          # nodes number
#SBATCH -n 1          # CPUs number (on all nodes) 
#SBATCH --exclusive
#SBATCH -q qos_cpu-t3
#SBATCH -o SPEC.eo%j   #
#SBATCH -e SPEC.eo%j   #
#SBATCH -t 19:59:00    # time limit
#SBATCH --export=NONE
#SBATCH -A whl@cpu # put here you account/projet name

# job information
cat << EOF
------------------------------------------------------------------
Job submit on $SLURM_SUBMIT_HOST by $SLURM_JOB_USER
JobID=$SLURM_JOBID Running_Node=$SLURM_NODELIST
Node=$SLURM_JOB_NUM_NODES Task=$SLURM_NTASKS
------------------------------------------------------------------
EOF

path="/linkhome/rech/genlmd01/rces071/Github/spectrum_analysis/src/"
file="Spectra_flux.py"
export MONORUN="python"

module load miniforge/24.9.0

time ${MONORUN} ${path}'/'${file}
