#!/bin/bash
#SBATCH --job-name=spark-cluster
#SBATCH --account=siads699s23_class        # change to your account
#SBATCH --partition=standard
#SBATCH --nodes=6                # node count, change as needed
#SBATCH --ntasks-per-node=1      # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=36       # cpu-cores per task, change as needed
#SBATCH --mem=180g               # memory per node, change as needed
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE

# These modules are required. You may need to customize the module version
# depending on which cluster you are on.
module load spark/3.2.1 python/3.10.4 pyarrow/8.0.0

# Start the Spark instance.
./spark-start-algorhythms

# Source spark-env.sh to get useful env variables.
source ${HOME}/.spark-local/${SLURM_JOB_ID}/spark/conf/spark-env.sh

# file_choice to be passed to script to discover the data and do the transfmation. Default = 'all'
file_choice=${1:-all}

# export PYSPARK_PYTHON=/home/anilcm/myenv/bin/python

# Read the path to the virtual environment 'venv_path' from the 'virtual_env.sh'
source ../conf/virtual_env.sh

# Update PYTHONPATH to include virtual environment's site-packages so that dependencies are available
export PYTHONPATH="$venv_path:$PYTHONPATH"

# Customize the executor resources below to match resources requested above
# with an allowance for spark driver overhead. Also change the path to your spark job.
spark-submit --master ${SPARK_MASTER_URL} \
  --executor-cores 1 \
  --executor-memory 16G \
  --total-executor-cores 70 \
  --py-files /home/anilcm/capstone_project/algorhythms/data_engineering/code/dependencies.zip \
  ../6_pytorch_geometric_data_obj_generator.py "$file_choice"