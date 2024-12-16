#!/bin/bash
#SBATCH -J STRUCT
#SBATCH -N 1          # nodes number
#SBATCH -n 1          # CPUs number (on all nodes) 
#SBATCH -q qos_cpu-dev
#SBATCH --exclusive           
#SBATCH -o STRUCT.eo%j   #
#SBATCH -e STRUCT.eo%j   #
#SBATCH -t 01:59:00    # time limit
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

path="/linkhome/rech/genlmd01/rces071/run/Github/spectrum_analysis/src/"
file="structure_functions.py"
export MONORUN="python"

module load miniforge/24.9.0

time ${MONORUN} ${path}'/'${file}
