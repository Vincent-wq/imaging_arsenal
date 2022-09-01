#!/bin/bash
#
#SBATCH -J template_vince
#SBATCH --time=168:00:00
#SBATCH --account=rpp-aevans-ab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=125G
# Outputs ----------------------------------
#SBATCH -o %x-%A-%a_%j.out
#SBATCH -e %x-%A-%a_%j.err
#SBATCH --mail-user=vincent.w.qing@gmail.com
#SBATCH --mail-type=ALL
# ------------------------------------------

DATA_DIR=(${@:1:1})
CON_IMG_DIR=(${@:2:1})

# in container
CODES_DIR=/home/vincentq/scratch/src

singularity exec --home /home/neuro --cleanenv \
        -B ${DATA_DIR}:/data \
        -B ${CODES_DIR}:/codes \
        ${CON_IMG_DIR} /codes/antsTmp2.sh
