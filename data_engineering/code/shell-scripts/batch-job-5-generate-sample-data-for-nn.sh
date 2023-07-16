#!/bin/bash
#SBATCH --job-name=spark-cluster
#SBATCH --account=siads699s23_class        # change to your account
#SBATCH --partition=standard
#SBATCH --nodes=3                # node count, change as needed
#SBATCH --ntasks-per-node=1      # do not change, leave as 1 task per node
#SBATCH --cpus-per-task=36       # cpu-cores per task, change as needed
#SBATCH --mem=180g               # memory per node, change as needed
#SBATCH --time=00:60:00
#SBATCH --mail-type=NONE

# These modules are required. You may need to customize the module version
# depending on which cluster you are on.
module load spark/3.2.1 python/3.10.4 pyarrow/8.0.0

# Start the Spark instance.
./spark-start-algorhythms

# Source spark-env.sh to get useful env variables.
source ${HOME}/.spark-local/${SLURM_JOB_ID}/spark/conf/spark-env.sh

# Sample size can be passed to the shell script. If none, 100,000 will be passed
sample_size=${1:-100000}

# Customize the executor resources below to match resources requested above
# with an allowance for spark driver overhead. Also change the path to your spark job.
spark-submit --master ${SPARK_MASTER_URL} \
  --executor-cores 1 \
  --executor-memory 5G \
  --total-executor-cores 70 \
  --py-files /home/anilcm/capstone_project/algorhythms/data_engineering/code/dependencies.zip \
  ../5_generate_sample_data_for_neuralnetwork.py "$sample_size"