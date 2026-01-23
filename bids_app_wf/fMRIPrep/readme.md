# fMRIPrep preprocessing
This is the fMRIPrep (version 20.2.7) scripts for slurm version for structrual and functional MRI preprocessing.

## example codes: 
'''
./repo_mdd_ptsd/workflow/fMRIPrep/fmriprep_sub.sh MDD_PTSD Y repo_mdd_ptsd/workflow/fMRIPrep/proj-mddPtsd_subjVisit-list.csv 
'''
## Processing logs
Processing subject MR224342 failure due to limited FOV of T1.

## Processing details 
There are 3 scripts for fMRIPrep, including: 

   1. **Main script**, e.g. `fmriprep_sub.sh`:
      1. Function: 1) Preparing the working directories (following compute canada conventions, like all the data in `~/scratch`): 2) Create subject list and save it as a file for latter use; 3) Download the templates (with TemplateFlow) for fMRIPrep; 4) Submit the computing task to compute canada with `sbatch`;
      2. Input: 1) the name of dataset you would like to preprocess; 2) Clean running flag; 3) Running list; 
      3. Output: 1) Standard [fMRIPrep outputs](https://fmriprep.org/en/stable/outputs.html); 2) Running logs.
   2. **Computing node script**, e.g. `fmriprep_sub.slurm` (the script will be submitted to the slurm cluster to run on each of the computing nodes);
      1. Function: The working horse of calling fMRIPrep singularity containner to run the preprocessing on computing node of the cluster: 1) Preparing the working environment for singularity containers (you will need a valid freesurfer liscence for this, which will be mounted in the singularity container as well); 2) Running the preprocessing with singularity container; 3) Writing log file.
      2. Input: The name of dataset you would like to preprocess;
      3. Output: 1) Standard [fMRIPrep outputs](https://fmriprep.org/en/stable/outputs.html) from each computing node; 2) Running log for each computing task.
   3. **Results collection script**, e.g. `fmriprep_collect.format` (Organize the preprocessed results and logs, prepare for data transfer).
      1. Function: Orgnizing and zipping the fMRIPrep ourputs and logs for data transfer:
      2. Input: The name of dataset you would like to preprocess;
      3. Output: ziped fMRIPrep results/freesurfer results/logs.
 
## Use example (take PPMI as use case):
1. Give execution to the **main script**: e.g. `chmod +x ET_biomarker/scripts/fmriprep/fmriprep_anat.sh`;
2. Run the **main code**: e.g. `./ET_biomarker/scripts/fmriprep/fmriprep_anat.sh PPMI`;
3. After the preprocessing finished, run the **results collection code**, e.g. `./ET_biomarker/scripts/fmriprep/fmriprep_anat.format PPMI`.
4. Data ready for further analysis!!! You can find  `~/scratch/res/PPMI_fmriprep_anat_${FMRIPREP_VER}.tar.gz` (the fMRIPrep results), `~/scratch/res/PPMI_fmriprep_anat_freesurfer_${FMRIPREP_VER}.tar.gz` (the freesurfer results) and `PPMI_fmriprep_anat_log.tar.gz` (log file).

## Reminder
  0. If there is any problems that you have encountered when trying to resuse these scripts, just open an issue and I will try my best to help.
  1. If there are any running errors (indicated from slurm logs, ***.err), try to rerun the preprocessing for the single subject with more computing resources (cores/RAM/computing time), it works for most of time, if not, you need to check the data quality;
  2. Usually, the computing node does not have access to the Internet for security reasons, and make sure you have downloaded the templates with TemplateFlow before submitting the computing tasks.