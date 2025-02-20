#!/bin/bash
# Execute run for a number of files
#

listfiles="../infos/files_FIR1k.txt"
#listfiles="../infos/files_IHOPNW.txt"
while IFS= read -r varname; do
    printf '%s\n' "$varname"
    sbatch run_figures.sh $varname
done < "$listfiles"
