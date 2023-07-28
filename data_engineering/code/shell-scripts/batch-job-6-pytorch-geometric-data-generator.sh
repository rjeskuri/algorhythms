#!/bin/bash
#SBATCH --job-name=pytorch-geometric-data-gen
#SBATCH --account=siads699s23_class        # change to your account
#SBATCH --partition=largemem
#SBATCH --nodes=1                # node count, change as needed
#SBATCH --ntasks-per-node=1      # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=1       # cpu-cores per task, change as needed
#SBATCH --mem=1000g               # memory per node, change as needed
#SBATCH --time=05:00:00
#SBATCH --mail-type=NONE

# These modules are required. You may need to customize the module version
# depending on which cluster you are on.
module load spark/3.2.1 python/3.10.4 pyarrow/8.0.0

# Read the path to the virtual environment 'venv_path' from the 'virtual_env.sh'
source ../conf/virtual_env.sh

# Update PYTHONPATH to include virtual environment's site-packages so that dependencies are available
export PYTHONPATH="$venv_path:$PYTHONPATH"

python ../6_pytorch_geometric_data_obj_generator.py