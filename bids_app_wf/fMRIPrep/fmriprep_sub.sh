#!/bin/bash
if [ "$#" -ne 5 ]; then
DATA_NAME=(${@:1:1})
CLEAN_RUN_FLAG=(${@:2:1})
fMRIPREP_mode=(${@:3:1})
RUN_LIST=(${@:4:1})
echo "Rerunning list of subjuects in " ${RUN_LIST}
RUNNING_TAB="Y"
else
DATA_NAME=(${@:1:1})
CLEAN_RUN_FLAG=(${@:2:1})
fMRIPREP_mode=(${@:3:1})
SUB_ID=(${@:4:1})
SES_ID=(${@:5:1})
echo 'Preprocessing subj: ' ${SUB_ID} ', ses' ${SES_ID} "with fMRIPrep mode" ${fMRIPREP_mode}
RUNNING_TAB="N"
fi
echo ${DATA_NAME} with fMRIPrep mode ${fMRIPREP_mode}

#data
WD_DIR=${HOME}/scratch
DATA_DIR=${WD_DIR}/${DATA_NAME}
DICOM_DIR=${DATA_DIR}/dicom
BIDS_DIR=${DATA_DIR}/bids

#codes
CODE_DIR=${WD_DIR}/repo_${DATA_NAME}/workflow/fMRIPrep # change according to project
CODE_SLURM=${CODE_DIR}/fmriprep_sub.slurm
CODE_COLLECT=${CODE_DIR}/fmriprep_collect.format
TEMPLATEFLOW_HOST_HOME=${WD_DIR}/templateflow

FMRIPREP_VER=23.2.1
# updating 23.2.1...
# 22.1.1: ERROR: Label BA1_exvivo does not exist in SUBJECTS_DIR fsaverage! fixed in this version.
# 20.2.7 run OK

#logs
LOG_FILE=${WD_DIR}/${DATA_NAME}_fmriprep-${FMRIPREP_VER}.log

# +x for codes and remove previous logs
chmod +x ${CODE_SLURM}
chmod +x ${CODE_COLLECT}

# check templateflow
if [ -d ${TEMPLATEFLOW_HOST_HOME} ];then
	echo "Templateflow dir already exists!"
else
	mkdir -p ${TEMPLATEFLOW_HOST_HOME}
	python -c "from templateflow import api; api.get('MNI152NLin2009cAsym')"
	python -c "from templateflow import api; api.get('OASIS30ANTs')"
fi


# Folder cleaning
if [ ${CLEAN_RUN_FLAG} == 'Y' ];then

# create session outputs
for SES_ in $(sed -n '/sub/s/^.*,\([0-9]*\).*$/\1/p' ${RUN_LIST} | sort -u)
do
echo "creating folders for session "$SES_
FMRIPREP_DIR=${DATA_DIR}/derivatives/fmriprep_ses-${SES_}_${FMRIPREP_VER}
if [ -d ${FMRIPREP_DIR} ];then
echo "cleaning folder for session "$SES_
rm -rf ${FMRIPREP_DIR}
fi
echo "creating new folder for session "$SES_
mkdir -p ${FMRIPREP_DIR}
done

# clearing logs
LOG_DIR=${DATA_DIR}_fmriprep-${FMRIPREP_VER}_log
if [ -d ${LOG_DIR} ];then
  rm -rf ${LOG_DIR}
fi
mkdir -p ${LOG_DIR}
fi

if [ ${RUNNING_TAB} == 'Y' ];then
while read line; do
    # Do what you want to $name
    SUB_ID_STR="$(cut -d',' -f1 <<<${line})"
    SUB_ID="$(cut -d'-' -f2 <<<${SUB_ID_STR})"
    SES_ID="$(cut -d',' -f2 <<<${line})"
    echo 'running subj: ' ${SUB_ID} ', ses' ${SES_ID}
    sbatch ${CODE_SLURM} ${DATA_NAME} ${FMRIPREP_VER} ${fMRIPREP_mode} ${SUB_ID} ${SES_ID} >> ${LOG_FILE}
    echo ${SUB_ID} ${SES_ID} >> ${LOG_FILE}
done < ${RUN_LIST}

else
echo 'running subj: ' ${SUB_ID} ', ses' ${SES_ID}
sbatch --containall ${CODE_SLURM} ${DATA_NAME} ${FMRIPREP_VER} ${fMRIPREP_mode} ${SUB_ID} ${SES_ID} >> ${LOG_FILE}
fi

echo "fmriprep job submitted!"
