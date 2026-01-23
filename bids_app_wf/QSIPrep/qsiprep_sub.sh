#!/bin/bash
if [ "$#" -ne 5 ]; then
DATA_NAME=(${@:1:1})
CLEAN_RUN_FLAG=(${@:2:1})
RUN_LIST=(${@:3:1})
PIPELINE_MODE=(${@:4:1})
echo "Rerunning list of subjuects in " ${RUN_LIST}
RUNNING_TAB="Y"
else
DATA_NAME=(${@:1:1})
CLEAN_RUN_FLAG=(${@:2:1})
SUB_ID=(${@:3:1})
SES_ID=(${@:4:1})
PIPELINE_MODE=(${@:5:1})
echo 'QSIPrep Preprocessing subj: ' ${SUB_ID} ', ses' ${SES_ID}
RUNNING_TAB="N"
fi
echo ${DATA_NAME} ${CLEAN_RUN_FLAG} ${RUNNING_TAB} ${PIPELINE_MODE}

#data
WD_DIR=${HOME}/scratch
DATA_DIR=${WD_DIR}/${DATA_NAME}
DICOM_DIR=${DATA_DIR}/dicom
BIDS_DIR=${DATA_DIR}/bids

#codes
CODE_DIR=${WD_DIR}/repo_${DATA_NAME}/workflow/QSIPrep # change according to project
CODE_SLURM=${CODE_DIR}/qsiprep_sub.slurm
CODE_COLLECT=${CODE_DIR}/qsiprep_collect.format

#
PIPELINE_VER=0.16.1

#logs
LOG_FILE=${WD_DIR}/${DATA_NAME}_QSIPrep-${PIPELINE_VER}.log

# +x for codes and remove previous logs
chmod +x ${CODE_SLURM}
chmod +x ${CODE_COLLECT}

# Folder cleaning
if [[ ${CLEAN_RUN_FLAG} == "Y" ]]; then
echo "running clreaning: "${CLEAN_RUN_FLAG}
# create session outputs 
if [[ ${RUNNING_TAB} == "Y" ]]; then
echo "processing list of subjects: "${RUNNING_TAB}
for SES_ in $(sed -n 's/^.*,\([0-9]*\).*$/\1/p' ${RUN_LIST} | sort -u)
do
OUT_DIR=${DATA_DIR}/derivatives/qisprep_ses-${SES_}_${PIPELINE_VER}_${PIPELINE_MODE}
echo "creating folders for session "$OUT_DIR
if [[-d ${OUT_DIR} ]]; then
echo "cleaning folder for session "$SES_
rm -rf ${OUT_DIR}
fi
echo "creating new folder for session "$SES_
mkdir -p ${OUT_DIR}
done

else
echo "creating folders for session "$SES_ID
OUT_DIR=${DATA_DIR}/derivatives/qsiprep_ses-${SES_ID}_${PIPELINE_VER}_${PIPELINE_MODE}
if [[ -d ${OUT_DIR} ]];then
echo "cleaning folder for session "$SES_ID
rm -rf ${OUT_DIR}
fi
echo "creating new folder for session "$SES_ID
mkdir -p ${OUT_DIR}
fi

# clearing logs
LOG_DIR=${DATA_DIR}_QSIPrep_${PIPELINE_VER}_log
if [[ -d ${LOG_DIR} ]];then
  rm -rf ${LOG_DIR}
fi
mkdir -p ${LOG_DIR}

fi

if [[ ${RUNNING_TAB} == "Y" ]];then
while read line; do
    # Do what you want to $name
    SUB_ID_STR="$(cut -d',' -f1 <<<${line})"
    SUB_ID="$(cut -d'-' -f2 <<<${SUB_ID_STR})"
    SES_ID="$(cut -d',' -f2 <<<${line})"
    echo 'running subj: ' ${SUB_ID} ', ses' ${SES_ID}
    sbatch ${CODE_SLURM} ${DATA_NAME} ${PIPELINE_VER} ${SUB_ID} ${SES_ID} ${PIPELINE_MODE} >> ${LOG_FILE}
done < ${RUN_LIST}

else
echo 'running subj: ' ${SUB_ID} ', ses' ${SES_ID}
sbatch ${CODE_SLURM} ${DATA_NAME} ${PIPELINE_VER} ${SUB_ID} ${SES_ID} ${PIPELINE_MODE} >> ${LOG_FILE}
fi

echo "QSIPrep job submitted!"
