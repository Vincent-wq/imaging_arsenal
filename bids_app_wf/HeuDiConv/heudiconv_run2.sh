#!/bin/bash
DATA_NAME=(${@:1:1})
echo ${DATA_NAME}
hpc_system=(${@:2:1})
echo ${hpc_system}

if [ ${hpc_system} == 'slurm' ]; then
# working dir for local machine
WD_DIR="${HOME}/scratch"

if [[ "$DATA_NAME" == *"control"* ]]; then
    repo_string="${DATA_NAME%_control}"
    CODE_DIR="${WD_DIR}/repo_${repo_string}/workflow/HeuDiConv"
    #Select the proper heuristic file
    HEURISTIC_FILE="Heuristics_${repo_string}_all.py"
else
    CODE_DIR="${WD_DIR}/repo_${DATA_NAME}/workflow/HeuDiConv"
    HEURISTIC_FILE="Heuristics_${DATA_NAME}_all.py"
fi

else
# working dir for cluster
WD_DIR="CLUSTER_WD_DIR"
CODE_DIR="CLUSTER_CODE_DIR"
fi 

# basic env and software
HEUDICONV_VERSION=0.9.0
#SUB_LIST=${CODE_DIR}/${DATA_NAME}_control_subjects.list
SUB_LIST=${CODE_DIR}/${DATA_NAME}_subjects.list # for ocd wave subjects 

#data dirs
DICOM_DIR=${WD_DIR}/${DATA_NAME}/dicom
BIDS_DIR=${WD_DIR}/${DATA_NAME}/bids
INFO_DIR=${WD_DIR}/${DATA_NAME}/conv_info
BACKUP_DIR=${WD_DIR}/${DATA_NAME}/backup

#logging
LOG_DIR=${WD_DIR}/logs/heudiconv
LOG_FILE_r2=${DATA_NAME}_heudiconv_run2.log

chmod +x ${CODE_DIR}/${HEURISTIC_FILE}
# Get total number of subjects
N_SUB=$(cat ${SUB_LIST}|wc -l )

# submit subject conversion batch job
if [ ${hpc_system} == 'slurm' ]; then
    # SLURM convention logs
    # running conversion
    CODE_FILE=${CODE_DIR}/heudiconv_run2.slurm
    echo "running for "${N_SUB}" subjects"
    #N_SUB=1 # for single subject test purpose
    sbatch --array=1-${N_SUB} ${CODE_FILE} ${DATA_NAME} ${HEURISTIC_FILE} ${HEUDICONV_VERSION} ${SUB_LIST} ${WD_DIR} >> ${LOG_FILE_r2}
    echo "SLURM job submitted!"
fi

