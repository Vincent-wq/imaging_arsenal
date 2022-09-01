#!/bin/bash

DATASET=/data
TEMPLATE=${DATASET}/template
mkdir -p ${TEMPLATE}
cd ${TEMPLATE}

${ANTSPATH}/antsMultivariateTemplateConstruction2.sh -d 3 -o T_ -c 2 -i 4 -f 6x4x2x1 -g .25 -n 0 -r 1 -m CC -t SyN -z ${TEMPLATE}T_template0.nii.gz ${DATASET}/*_desc-masked_T1w.nii.gz
