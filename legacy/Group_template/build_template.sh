#!/bin/bash
DATA_NAME=(${@:1:1})
echo ${DATA_NAME}

SCRATCH_DIR=/home/vincentq/scratch
CODE_DIR=${SCRATCH_DIR}/src
CODE_SLURM=${CODE_DIR}/build_template.slurm
CON_IMG_DIR=${SCRATCH_DIR}/container_images/vincent_env_v0.0.simg
LOG_DIR=$DATA_NAME/build_template.log

sbatch ${CODE_SLURM} ${DATA_NAME} ${CON_IMG_DIR} >> ${LOG_DIR}


