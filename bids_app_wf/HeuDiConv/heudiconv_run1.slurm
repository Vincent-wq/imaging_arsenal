#!/bin/bash
#SBATCH --job-name=heudi_vinc_r1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
# Outputs ----------------------------------
#SBATCH -o %x-%A-%a_%j.out
#SBATCH -e %x-%A-%a_%j.err

DATA_NAME=(${@:1:1})
HEUDICONV_VERSION=(${@:2:1})
SUB_LIST=(${@:3:1})
WD_DIR=(${@:4:1})
echo converting ${DATA_NAME} with subjects list:${SUB_LIST} using HeuDiConv version=${HEUDICONV_VERSION}...

# dataset dir
echo "Current Directory: " ${WD_DIR}

# codes, envs and containers
CON_IMG=${WD_DIR}/container_images/heudiconv_${HEUDICONV_VERSION}.sif

# singularity folders
SINGULARITY_MNT_DIR=/scratch
SINGULARITY_DATA_DIR=${SINGULARITY_MNT_DIR}/${DATA_NAME}/dicom
SINGULARITY_OUT_DIR=${SINGULARITY_MNT_DIR}/${DATA_NAME}/bids

echo "Starting task ${SLURM_ARRAY_TASK_ID}"
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${SUB_LIST} )
echo ${DIR}
subject_id=$(echo ${DIR} | grep -oE "[^/]+$")
echo ${DATA_NAME} ${subject_id}

#for _ses_dir in ${DIR}/*; do
#ses_id=$(echo ${_ses_dir} | grep -oE "[^/]+$")
#echo "screening session: "${ses_id}
singularity run --cleanenv -B ${WD_DIR}:${SINGULARITY_MNT_DIR} ${CON_IMG} \
-d ${SINGULARITY_DATA_DIR}/{subject}/*/*.dcm \
-s ${subject_id} -c none \
-ss "0" \
-f convertall \
-o ${SINGULARITY_OUT_DIR} \
--overwrite
#done

echo "Heudiconv Run1 finishted!"
