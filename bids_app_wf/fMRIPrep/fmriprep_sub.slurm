#!/bin/bash
#
#SBATCH -J fMRIPrep_vin
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2G
# Outputs ----------------------------------
#SBATCH -o %x-%A-%a_%j.out
#SBATCH -e %x-%A-%a_%j.err
# ------------------------------------------

DATA_NAME=(${@:1:1})
PIPELINE_VER=(${@:2:1})
fMRIPREP_mode=(${@:3:1})
SUB_ID=(${@:4:1})
SES_ID=(${@:5:1})

#dirs
WD_DIR=${HOME}/scratch
DATA_DIR=${WD_DIR}/${DATA_NAME}
BIDS_DIR=${DATA_DIR}/bids
CON_IMG=${WD_DIR}/container_images/fmriprep_${PIPELINE_VER}.sif
FMRIPREP_DIR=${DATA_DIR}/derivatives/fmriprep_ses-${SES_ID}_${PIPELINE_VER}

# Need to change according to projects
CODE_DIR=${WD_DIR}/repo_${DATA_NAME}/workflow/fMRIPrep # change according to project
BIDS_FILTER_FILE="func_ses-${SES_ID}.json"
BIDS_FILTER=/codes/${BIDS_FILTER_FILE}
echo ${SUB_ID}***${BIDS_FILTER_FILE}

FMRIPREP_HOME=fmriprep_home_${SUB_ID}_${SES_ID}
WORK_DIR=${DATA_DIR}_work_${SUB_ID}_${SES_ID}
echo "Processing: sub-${SUB_ID}, session ${SES_ID} with home dir: ${FMRIPREP_HOME}"
mkdir -p ${FMRIPREP_HOME}
mkdir -p ${WORK_DIR}

LOCAL_FREESURFER_DIR="${FMRIPREP_DIR}/freesurfer-7.2.0"
mkdir -p ${LOCAL_FREESURFER_DIR}

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME=$HOME/scratch/templateflow
FMRIPREP_HOST_CACHE=$FMRIPREP_HOME/.cache/fmriprep
mkdir -p ${FMRIPREP_HOST_CACHE}

# Make sure FS_LICENSE is defined in the container.
mkdir -p $FMRIPREP_HOME/.freesurfer
export SINGULARITYENV_FS_LICENSE=$FMRIPREP_HOME/.freesurfer/license.txt
cp container_images/license.txt ${SINGULARITYENV_FS_LICENSE}

# Designate a templateflow bind-mount point
export SINGULARITYENV_TEMPLATEFLOW_HOME="/templateflow"
SINGULARITY_CMD="singularity run --containall -B ${FMRIPREP_HOME}:/home/fmriprep --home /home/fmriprep --cleanenv \
-B ${BIDS_DIR}:/data:ro \
-B ${FMRIPREP_DIR}:/output \
-B ${CODE_DIR}:/codes \
-B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} \
-B ${WORK_DIR}:/work \
-B ${LOCAL_FREESURFER_DIR}:/fsdir ${CON_IMG}"

# Remove IsRunniGng files from FreeSurfer
find ${LOCAL_FREESURFER_DIR}/sub-$SUB_ID/ -name "*IsRunning*" -type f -delete

if [ ${fMRIPREP_mode} == 'anat' ];then

cmd="${SINGULARITY_CMD} /data /output participant -w /work \
--participant-label $SUB_ID \
--anat-only \
--ignore t2w \
--output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 \
--cifti-out 91k \
--bids-filter-file ${BIDS_FILTER} \
--fs-subjects-dir /fsdir \
--fs-license-file /home/fmriprep/.freesurfer/license.txt \
--return-all-components \
--skip_bids_validation --write-graph --notrack -vv"

else
cmd="${SINGULARITY_CMD} /data /output participant -w /work \
--participant-label $SUB_ID \
--ignore t2w \
--output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 \
--cifti-out 91k \
--bids-filter-file ${BIDS_FILTER} \
--fs-subjects-dir /fsdir \
--fs-license-file /home/fmriprep/.freesurfer/license.txt \
--return-all-components \
--skip_bids_validation --write-graph --notrack -vv"
fi
# remove --use-aroma, since it is deprecated in 24.0
# Compose the command line

# --nprocs 4 --mem 7990
#--fd-spike-threshold --dvars-spike-threshold \
# --bids-filter-file ${BIDS_FILTER} --anat-only --ignore t2w 
# Setup done, run the command
#echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
unset PYTHONPATH
eval $cmd
exitcode=$?

# Output results to a table
touch ${SLURM_JOB_NAME}.tsv
echo "${SUB_ID}    ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
      >> ${SLURM_JOB_NAME}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}
rm -rf ${FMRIPREP_HOME}
rm -rf ${WORK_DIR}
exit ${exitcode}
