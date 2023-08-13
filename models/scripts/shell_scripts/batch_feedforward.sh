#!/bin/bash
#SBATCH --job-name=pytorch-geometric-data-gen
#SBATCH --account=siads699s23_class        # change to your account
#SBATCH --partition=standard
#SBATCH --nodes=1                # node count, change as needed
#SBATCH --ntasks-per-node=1      # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=8       # cpu-cores per task, change as needed
#SBATCH --mem=180g               # memory per node, change as needed
#SBATCH --time=05:00:00
#SBATCH --mail-type=NONE

# These modules are required. You may need to customize the module version
# depending on which cluster you are on.
module load python/3.10.4 pyarrow/8.0.0

# Pass the folder name within 'saved_files/data_representations' as 'data_version_input'
data_version_input="$1"
embedding_folder_file="$2"
num_epochs="$3"

python -u ../feed_forward_model.py "$data_version_input" "$embedding_folder_file" "$num_epochs" >> slurm-${SLURM_JOB_ID}.out 2>&1
