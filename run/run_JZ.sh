#!/bin/bash
#SBATCH -J SPECI
#SBATCH -N 1          # nodes number
#SBATCH -n 1          # CPUs number (on all nodes) 
##SBATCH -q qos_cpu-t3
##SBATCH --partition=cpu_p1
#SBATCH --partition=prepost
#SBATCH -o SPEC.eo%j   #
#SBATCH -e SPEC.eo%j   #
#SBATCH -t 04:59:00    # time limit
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

# Name of the file
file=$1
echo $file

path="/linkhome/rech/genlmd01/rces071/Github/spectrum_analysis/src/"
filepy="spectral_analysis.py"
export MONORUN="python"

module load miniforge/24.9.0
#ln -sf ../infos/info_run_JZ.txt ../infos/info_run.txt

time ${MONORUN} ${path}'/'${filepy} ${file} 
