#!/bin/bash
#SBATCH --job-name=heudi_vinc_r2
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
# Outputs ----------------------------------
#SBATCH -o %x-%A-%a_%j.out
#SBATCH -e %x-%A-%a_%j.err

# usage:
#sbatch heudiconv_slurm_run2.sh ${WD_NAME} ${STUDY_NAME} ${SEARCH_LV} ${HEURISTIC_FILE} >> ${LOG_FILE}_run2.log
DATA_NAME=(${@:1:1})
HEURISTIC_FILE=(${@:2:1})
HEUDICONV_VERSION=(${@:3:1})
SUB_LIST=(${@:4:1})
WD_DIR=(${@:5:1})

CON_IMG=${WD_DIR}/container_images/heudiconv_${HEUDICONV_VERSION}.sif
echo "In "${WD_DIR}", converting: " ${DATA_NAME} " subjects: " ${SUB_LIST}  " with contatiner: " ${CON_IMG} "!"

# prepare folders
#WD_DIR=${HOME}/"scratch"
DICOM_DIR=${WD_DIR}/${DATA_NAME}/dicom
BIDS_DIR=${WD_DIR}/${DATA_NAME}_BIDS

# singularity folders
SINGULARITY_MNT_DIR=/scratch
SINGULARITY_DATA_DIR=${SINGULARITY_MNT_DIR}/${DATA_NAME}/dicom
SINGULARITY_OUT_DIR=${SINGULARITY_MNT_DIR}/${DATA_NAME}/bids
#SINGULARITY_HEURISTICS=${SINGULARITY_MNT_DIR}/repo_${DATA_NAME}/workflow/HeuDiConv/${HEURISTIC_FILE}
echo " heuristics file in container: " ${SINGULARITY_HEURISTICS} 

if [[ "$DATA_NAME" == *"control"* ]]; then
    repo_string="${DATA_NAME%_control}"
    CODE_DIR="${WD_DIR}/repo_${repo_string}/workflow/HeuDiConv"
    SINGULARITY_HEURISTICS=${SINGULARITY_MNT_DIR}/repo_${repo_string}/workflow/HeuDiConv/${HEURISTIC_FILE}
else
    CODE_DIR="${WD_DIR}/repo_${DATA_NAME}/workflow/HeuDiConv"
    SINGULARITY_HEURISTICS=${SINGULARITY_MNT_DIR}/repo_${DATA_NAME}/workflow/HeuDiConv/${HEURISTIC_FILE}
fi

# run heudiconv at subject level.
echo "Starting task ${SLURM_ARRAY_TASK_ID}"
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${SUB_LIST})
echo ${DIR}
subject_id=$(echo ${DIR} | grep -oE "[^/]+$")

#for _ses_dir in ${DIR}/*; do
#ses_id=$(echo ${_ses_dir} | grep -oE "[^/]+$")

#echo "converting session: "${ses_id}
singularity run --cleanenv -B ${WD_DIR}:${SINGULARITY_MNT_DIR} ${CON_IMG} \
-d ${SINGULARITY_DATA_DIR}/{subject}/*/*.dcm \
-s ${subject_id} \
-ss "0" \
-f ${SINGULARITY_HEURISTICS} \
-c dcm2niix -b --overwrite --minmeta \
-o ${SINGULARITY_OUT_DIR} \
--overwrite
#-ss ${ses_id} \
#done

echo "Heudiconv Run2 finishted, conversion complete for subject: "${subject_id}"!"
