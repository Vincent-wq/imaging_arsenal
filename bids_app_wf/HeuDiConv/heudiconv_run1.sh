#!/usr/bin/bash
DATA_NAME=(${@:1:1})
echo ${DATA_NAME}
hpc_system=(${@:2:1})
echo ${hpc_system}
CHECK_DIR=(${@:3:1})
echo ${CHECK_DIR}

if [ ${hpc_system} == 'slurm' ]; then
# working dir for local machine
WD_DIR="${HOME}/scratch"

if [[ "$DATA_NAME" == *"control"* ]]; then
    repo_string="${DATA_NAME%_control}"
    CODE_DIR="${WD_DIR}/repo_${repo_string}/workflow/HeuDiConv"
else
    CODE_DIR="${WD_DIR}/repo_${DATA_NAME}/workflow/HeuDiConv"
fi

else
# working dir for cluster
WD_DIR="CLUSTER_WD_DIR"
CODE_DIR="CLUSTER_CODE_DIR"
fi 

# basic env and software
HEUDICONV_VERSION=0.9.0
SEARCH_LV=1
SUB_LIST=${CODE_DIR}/${DATA_NAME}_subjects.list

#data
DICOM_DIR=${WD_DIR}/${DATA_NAME}/dicom
BIDS_DIR=${WD_DIR}/${DATA_NAME}/bids
INFO_DIR=${WD_DIR}/${DATA_NAME}/conv_info
BACKUP_DIR=${WD_DIR}/${DATA_NAME}/backup

#logging
LOG_DIR=${WD_DIR}/logs/heudiconv
LOG_FILE_prefix=${DATA_NAME}_heudiconv
LOG_FILE_r1=${LOG_FILE_prefix}_run1.log
LOG_FILE_r2=${LOG_FILE_prefix}_run2.log

# load modules on HPC
#module load singularity
RUN_ID=$(tail -c 9 ${LOG_FILE_r1})
if [ -z $RUN_ID ];then
  echo 'no previous run found...'
else
  echo "previous run $RUN_ID found, deleting logs..."
  rm -rf ${LOG_DIR}/vincentq_heudiconv_r1_*
fi

chmod +x ${CODE_DIR}/heudiconv_run2.sh
chmod +x ${CODE_DIR}/heudiconv_run1.format
chmod +x ${CODE_DIR}/heudiconv_run2.format

# get all subject dicom foldernames.
rm ${SUB_LIST}
find ${DICOM_DIR} -maxdepth ${SEARCH_LV} -mindepth ${SEARCH_LV} >> ${SUB_LIST}
N_SUB=$(cat ${SUB_LIST}|wc -l )
echo "Step1: subjects.list created!"

# folder check
if [ ${CHECK_DIR} == 'Y' ];then
if [ -d ${BIDS_DIR} ];then
  rm -rf ${BIDS_DIR}
  rm -rf ${BACKUP_DIR}/${DATA_NAME}_bids.zip
  mkdir -p ${BIDS_DIR}
  echo "BIDS folder already exists, cleared!"
else
  mkdir -p ${BIDS_DIR}
fi

if [ -d ${INFO_DIR} ];then
  rm -rf ${INFO_DIR}
  rm -rf ${BACKUP_DIR}/${DATA_NAME}_info.zip 
  mkdir -p ${INFO_DIR}
  echo "INFO_SUM folder already exists, cleared!"
else
  mkdir -p ${INFO_DIR}
fi

if [ -d ${LOG_DIR} ];then
  rm -rf ${LOG_DIR}/*
  echo "SLURM_LOG_OUT_DIR_run1 folder already exists, cleared!"
else
  mkdir -p ${LOG_DIR}
fi
else
echo "Folders check skipped!"
fi

# submit batch job
if [ ${hpc_system} == 'slurm' ]; then
    chmod +x ${CODE_DIR}/heudiconv_run1.slurm
    chmod +x ${CODE_DIR}/heudiconv_run2.slurm
    CODE_FILE=${CODE_DIR}/heudiconv_run1.slurm
    #N_SUB=1 # for single subject test purpose
    sbatch --array=1-${N_SUB} ${CODE_FILE} ${DATA_NAME} ${HEUDICONV_VERSION} ${SUB_LIST} ${WD_DIR} >> ${LOG_FILE_r1}
    echo "SLURM job submitted!"
else
    chmod +x ${CODE_DIR}/heudiconv_run1.sge
    chmod +x ${CODE_DIR}/heudiconv_run2.sge
    CODE_FILE=${CODE_DIR}/heudiconv_run1.sge
    #N_SUB=1 # for single subject test purpose
    qsub -t 1-${N_SUB} -q origami.q ${CODE_FILE} ${DATA_NAME} ${HEUDICONV_VERSION} ${SUB_LIST} ${WD_DIR} >> ${LOG_FILE_r1}
    echo "SGE job submitted!"
    
fi 
