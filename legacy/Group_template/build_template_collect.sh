#!/bin/bash
DATA_NAME=(${@:1:1})
OUT_DIR=(${@:2:1})
echo ${DATA_NAME}
echo ${OUT_DIR}
#OUT_DIR=group_template

mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/ET-ind $OUT_DIR/ET-mni $OUT_DIR/PD-ind $OUT_DIR/PD-mni $OUT_DIR/NC-ind $OUT_DIR/NC-mni

mv ${DATA_NAME}/ET/individual/template $OUT_DIR/ET-ind
mv ${DATA_NAME}/ET/mni2009cAsym/template $OUT_DIR/ET-mni
mv ${DATA_NAME}/PD/individual/template $OUT_DIR/PD-ind
mv ${DATA_NAME}/PD/mni2009cAsym/template $OUT_DIR/PD-mni
mv ${DATA_NAME}/NC/individual/template $OUT_DIR/NC-ind
mv ${DATA_NAME}/NC/mni2009cAsym/template $OUT_DIR/NC-mni
mv template_vince* $OUT_DIR

tar -czvf ${OUT_DIR}.tar.gz ${OUT_DIR}
