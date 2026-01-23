#!/usr/bin/bash
FS_RES_DIR=(${@:1:1})
OUT_DIR=(${@:2:1})
# Use notes:
# Make sure the bash and freesurfer dirs are correct
# run sed -i -e 's/\r$//' fs_stats2table.sh if new lines are added in win OS.
# usage: ./fs_stats2table.sh <FS_SUBJECT_DIR> <OUT_PUT_DIR>

echo "This bash script will create table from ?.stats files"
echo "Written by Vincent"
echo "@SMHC"
echo "Please check the number of output files, it should be 34 in total."
echo "04/05/2023\n"

# run in python 2.7 env
# run in the folder of freesurfer outputs
export FREESURFER_HOME=/opt/freesurfer
#export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=${FS_RES_DIR}

list_with_fsdir=$(ls ${FS_RES_DIR} | tr "\n" " ")
list=$(echo ${list_with_fsdir})
echo "Subjects detected: "${list}
# put subject list here 
#list="sub-season101 sub-season102 sub-season103 sub-season104 sub-season105 sub-season106 sub-season107 sub-season108 sub-season109 sub-season110 sub-season111 sub-season112 sub-season114 sub-season116 sub-season117 sub-season119 sub-season120 sub-season121 sub-season122 sub-season123 sub-season124 sub-season125 sub-season126 sub-season127 sub-season128 sub-season129 sub-season130 sub-season131 sub-season132 sub-season133 sub-season134 sub-season135 sub-season136 sub-season137 sub-season138 sub-season139 sub-season202 sub-season204 sub-season206 sub-season207 sub-season208 sub-season211 sub-season212 sub-season213 sub-season214 sub-season215 sub-season216 sub-season217"

# create output dir
if [ -d ${OUT_DIR} ];then
	echo "Output dir already exists, remove old files!"
    rm -rf ${OUT_DIR}/*
else
	mkdir -p ${OUT_DIR}
fi

asegstats2table --subjects $list --meas volume --skip --statsfile wmparc.stats --all-segs --tablefile ${OUT_DIR}/wmparc_stats.txt
asegstats2table --subjects $list --meas volume --skip --tablefile ${OUT_DIR}/aseg_stats.txt
#ind space
aparcstats2table --subjects $list --hemi lh --meas volume --skip --tablefile ${OUT_DIR}/aparc_volume_lh.txt
aparcstats2table --subjects $list --hemi lh --meas thickness --skip --tablefile ${OUT_DIR}/aparc_thickness_lh.txt
aparcstats2table --subjects $list --hemi lh --meas area --skip --tablefile ${OUT_DIR}/aparc_area_lh.txt
aparcstats2table --subjects $list --hemi lh --meas meancurv --skip --tablefile ${OUT_DIR}/aparc_meancurv_lh.txt
aparcstats2table --subjects $list --hemi rh --meas volume --skip --tablefile ${OUT_DIR}/aparc_volume_rh.txt
aparcstats2table --subjects $list --hemi rh --meas thickness --skip --tablefile ${OUT_DIR}/aparc_thickness_rh.txt
aparcstats2table --subjects $list --hemi rh --meas area --skip --tablefile ${OUT_DIR}/aparc_area_rh.txt
aparcstats2table --subjects $list --hemi rh --meas meancurv --skip --tablefile ${OUT_DIR}/aparc_meancurv_rh.txt
# parc a2009s
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas volume --skip -t ${OUT_DIR}/lh.a2009s.volume.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas thickness --skip -t ${OUT_DIR}/lh.a2009s.thickness.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas area --skip -t ${OUT_DIR}/lh.a2009s.area.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas meancurv --skip -t ${OUT_DIR}/lh.a2009s.meancurv.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas volume --skip -t ${OUT_DIR}/rh.a2009s.volume.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas thickness --skip -t ${OUT_DIR}/rh.a2009s.thickness.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas area --skip -t ${OUT_DIR}/rh.a2009s.area.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas meancurv --skip -t ${OUT_DIR}/rh.a2009s.meancurv.txt
# DKTatlas
aparcstats2table --hemi lh --subjects $list --parc aparc.DKTatlas --meas volume --skip -t ${OUT_DIR}/lh.DKTatlas.volume.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.DKTatlas --meas thickness --skip -t ${OUT_DIR}/lh.DKTatlas.thickness.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.DKTatlas --meas area --skip -t ${OUT_DIR}/lh.DKTatlas.area.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.DKTatlas --meas meancurv --skip -t ${OUT_DIR}/lh.DKTatlas.meancurv.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.DKTatlas --meas volume --skip -t ${OUT_DIR}/rh.DKTatlas.volume.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.DKTatlas --meas thickness --skip -t ${OUT_DIR}/rh.DKTatlas.thickness.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.DKTatlas --meas area --skip -t ${OUT_DIR}/rh.DKTatlas.area.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.DKTatlas --meas meancurv --skip -t ${OUT_DIR}/rh.DKTatlas.meancurv.txt
# parc BA_exvivo
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas volume --skip -t ${OUT_DIR}/lh.BA_exvivo.volume.txt
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas thickness --skip -t ${OUT_DIR}/lh.BA_exvivo.thickness.txt
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas area --skip -t ${OUT_DIR}/lh.BA_exvivo.area.txt
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas meancurv --skip -t ${OUT_DIR}/lh.BA_exvivo.meancurv.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas volume --skip -t ${OUT_DIR}/rh.BA_exvivo.volume.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas thickness --skip -t ${OUT_DIR}/rh.BA_exvivo.thickness.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas area --skip -t ${OUT_DIR}/rh.BA_exvivo.area.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas meancurv --skip -t ${OUT_DIR}/rh.BA_exvivo.meancurv.txt
