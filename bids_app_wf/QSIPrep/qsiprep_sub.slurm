#!/bin/bash
#
#SBATCH -J qsi_vincent
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
# Outputs ----------------------------------
#SBATCH -o %x-%A-%a_%j.out
#SBATCH -e %x-%A-%a_%j.err
# ------------------------------------------

DATA_NAME=(${@:1:1})
PIPELINE_VER=(${@:2:1})
SUB_ID=(${@:3:1})
SES_ID=(${@:4:1})
PIPELINE_MODE=(${@:5:1})

#dirs
WD_DIR=${HOME}/scratch
DATA_DIR=${WD_DIR}/${DATA_NAME}
BIDS_DIR=${DATA_DIR}/bids
CON_IMG=${WD_DIR}/container_images/qsiprep_${PIPELINE_VER}.sif
PROCESSED_DIR=${DATA_DIR}/derivatives/qisprep_ses-${SES_ID}_${PIPELINE_VER}_${PIPELINE_MODE}
FS_DIR=${DATA_DIR}/derivatives/fmriprep_ses-${SES_ID}_22.1.1/freesurfer-7.2.0

# Need to change according to projects
CODE_DIR=${WD_DIR}/repo_${DATA_NAME}/workflow/QSIPrep # change according to project
BIDS_FILTER_FILE="dwi_ses-${SES_ID}.json"
# in container
BIDS_FILTER=/codes/${BIDS_FILTER_FILE}
echo ${SUB_ID}***${SES_ID}***${BIDS_FILTER_FILE}


PROC_HOME=qsiprep_home_${SUB_ID}_${SES_ID}
WORK_DIR=${DATA_DIR}_qsiprep-work_${SUB_ID}_${SES_ID}
echo "Processing: sub-${SUB_ID}, session ${SES_ID} with home dir: ${PROC_HOME} in mode ${PIPELINE_MODE}"  
mkdir -p ${PROC_HOME}
mkdir -p ${WORK_DIR}

#LOCAL_FREESURFER_DIR="${DATA_DIR}/derivatives/fmriprep_20.2.7/freesurfer-6.0.3"
#mkdir -p ${LOCAL_FREESURFER_DIR}

# Prepare some writeable bind-mount points.
IMG_HOST_CACHE=$PROC_HOME/.cache/qsiprep
mkdir -p ${IMG_HOST_CACHE}

# Make sure FS_LICENSE is defined in the container.
mkdir -p $PROC_HOME/.freesurfer
export SINGULARITYENV_FS_LICENSE=$PROC_HOME/.freesurfer/license.txt
cp ${HOME}/scratch/container_images/license.txt ${SINGULARITYENV_FS_LICENSE}

# Designate a templateflow bind-mount point
echo ${PROCESSED_DIR}
[ -d ${PROCESSED_DIR} ] && echo "Directory ${PROCESSED_DIR} exists." || echo "Error: Directory ${PROCESSED_DIR} does not exists."

SINGULARITY_CMD="singularity run --cleanenv \
    -B ${BIDS_DIR}:/data:ro \
    -B ${FS_DIR}:/freesurfer:ro \
    -B ${PROCESSED_DIR}:/output_dir \
    -B ${WORK_DIR}:/work \
    -B ${CODE_DIR}:/codes \
    -B ${SINGULARITYENV_FS_LICENSE}:/opt/freesurfer-6.0.0/license.txt ${CON_IMG}"

#     -B ${LOCAL_FREESURFER_DIR}:/freesurfer_directory:ro\
## Remove IsRunning files from FreeSurfer
#find ${LOCAL_FREESURFER_DIR}/sub-$SUB_ID/ -name "*IsRunning*" -type f -delete

echo "SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK " and SLURM_MEM_PER_NODE="$SLURM_MEM_PER_NODE 
# Compose the command line
cmd="${SINGULARITY_CMD} /data /output_dir participant \
--participant_label $SUB_ID \
-w /work \
--output-resolution 1.2 \
--output-space template \
--freesurfer-input /freesurfer \
--fs-license-file /opt/freesurfer-6.0.0/license.txt \
--recon_input  /output_dir \
--recon_spec ${PIPELINE_MODE} \
--skip_bids_validation \
--bids-filter-file ${BIDS_FILTER} \
--notrack --nthreads $SLURM_CPUS_PER_TASK \
"
# --recon_spec gqi_scalar_export.json \
# --fs-license-file /opt/freesurfer-6.0.0/license.txt \
# - --mem_mb $SLURM_MEM_PER_NODE  --nv
# Setup done, run the command
#echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
unset PYTHONPATH
eval $cmd
exitcode=$?

# Output results to a table
echo "${SUB_ID}    $SLURM_PROCID    ${exitcode}" \
      >> ${SLURM_JOB_NAME}.tsv
echo Finished tasks $SLURM_PROCID with exit code ${exitcode}
rm -rf ${PROC_HOME}
rm -rf ${WORK_DIR}
exit ${exitcode}

### test code

#####SBATCH --gres=gpu
