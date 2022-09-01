#! /usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author: Vincent (Qing Wang)
# @Date:   2020-7-30 12:00:00
"""
======================================================
xxxxxxx.
======================================================
1. Libs and functions
"""
import os
import sys
import nipype.interfaces.utility as util
import nipype.interfaces.ants as ants
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe  # pypeline engine

def DataGraberRun(SUB_LIST, DATA_DIR, TMPT):
    import nipype.interfaces.io as nio
    with open(SUB_LIST, 'r') as f_sub:
        sub_list = f_sub.readlines()
    sub_list = [x[0:-1] for x in sub_list]
    print('[--- ', len(sub_list), ' ---] subjects included: ', sub_list)
    ds = nio.DataGrabber(infields=['subject_id', 'subject_id'])
    ds.inputs.base_directory = DATA_DIR # database
    ds.inputs.template = TMPT # from cwd
    ds.inputs.subject_id = sub_list
    ds.inputs.sort_filelist = True
    res = ds.run()
    res_list = res.outputs.outfiles
    return res_list
def ants_reg(MV_IMG, FIX_IMG, PREFIX_STR, ):
    import time
    #ants reg
    t0=time.time()
    ants = ANTS()
    ants.inputs.dimension = 3
    ants.inputs.num_threads = 6
    ants.inputs.output_transform_prefix = PREFIX_STR
    ants.inputs.metric = ['CC']
    ants.inputs.fixed_image  = [FIX_IMG] #atlas_09_masked
    ants.inputs.moving_image = [MV_IMG] #t1_masked
    ants.inputs.metric_weight= [1.0]
    ants.inputs.radius = [4]
    ants.inputs.transformation_model = 'SyN'
    ants.inputs.gradient_step_length = 0.25
    ants.inputs.number_of_iterations = [50, 35, 20]
    ants.inputs.use_histogram_matching = True
    ants.inputs.mi_option = [32, 16000]
    ants.inputs.regularization = 'Gauss'
    ants.inputs.regularization_gradient_field_sigma = 3
    ants.inputs.regularization_deformation_field_sigma = 0
    ants.inputs.number_of_affine_iterations = [10000,10000,10000,10000,10000]
    ants.cmdline
    ants.run()
    t1=time.time()
    return 1
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Input of pamameters: ')
    parser.add_argument('--group',   type=str, default = 'PD')
    parser.add_argument('--data',   type=str, default = '/data')
    parser.add_argument('--output', type=str, default = '/output')
    parser.add_argument('--sub_list',   type=str, default = '/output/subjects.list')
    parser.add_argument('--nprocs', type=int, default = 2)
    args = parser.parse_args()
    return args
    
def main(STUDY_GRP, DATA_DIR, OUT_DIR, SUB_LIST, N_PROCESS):
    """Entry point"""
    #from niflow.nipype1.workflows.smri.ants import antsRegistrationTemplateBuildSingleIterationWF
    """1. input images and basic output dir"""
    out_dir = OUT_DIR
    if not os.path.exists(OUT_DIR):
        os.makedirs(out_dir)
    TMPT_t1='%s/anat/%s_desc-preproc_T1w.nii.gz'
    TMPT_t1_9c='%s/anat/%s_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
    #TMPT_t1_6a='%s/anat/%s_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz'
    TMPT_t1_mask='%s/anat/%s_desc-brain_mask.nii.gz'
    # subject list
    t1_list = DataGraberRun(SUB_LIST, DATA_DIR, TMPT_t1)
    print('Subject T1 files: ', t1_list)
    t1_9c_list = DataGraberRun(SUB_LIST, DATA_DIR, TMPT_t1_9c)
    print('Subject T1 (MNI2009c Space) files: ', t1_9c_list)
    t1_mask_list = DataGraberRun(SUB_LIST, DATA_DIR, TMPT_t1_mask)
    print('Subject T1  mask files: ', t1_mask_list)
    # template
    atlas_9c='/template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
    atlas_9c_mask='/template/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'

    """1. input images"""

from nipype.interfaces import fsl
from nipype.interfaces.ants import ANTS, ApplyTransforms,CreateJacobianDeterminantImage
import os,time

OUT_DIR='/output/sMRI/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
atlas_9c='/templateflow/atlas/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
atlas_9c_mask='/templateflow/atlas/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
atlas_09_masked='/output/sMRI/t1_MNI2009c_masked.nii.gz'
mask = fsl.ApplyMask(
    in_file=atlas_9c,
    out_file=atlas_09_masked,
    mask_file=atlas_9c_mask)
mask.run()

    str_sub = t1_list[i_sub]
    t1_masked = OUT_DIR+str_sub[23:-7] + '-masked.nii.gz'
    print('t1_masked: ',t1_masked)
    transform_prefix = t1_masked[13:-7]
    t1_deformed = transform_prefix+'_'+'space-MNI2009c.nii.gz'
    print('t1_deformed: ',t1_deformed)
    str_output_transform_prefix = OUT_DIR+t1_deformed[0:-7]+'_'
    print('str_output_transform_prefix: ',str_output_transform_prefix)
    t1_warp   = str_output_transform_prefix+'Warp.nii.gz'
    t1_affine = str_output_transform_prefix+'Affine.txt'
    t1_deformed_jacobian = str_output_transform_prefix+'Jacobian.nii.gz'
    print('processing: ',str_sub, ':\n' )
    #apply mask
    mask = fsl.ApplyMask(
        in_file=t1_list[i_sub],
        out_file=t1_masked,
        mask_file=t1_mask_list[i_sub])
    mask.run()
    
    
    print('reg takes: ', t1-t0)
    # apply deformation
    at1 = ApplyTransforms()
    at1.inputs.dimension = 3
    at1.inputs.input_image = t1_masked
    at1.inputs.reference_image = atlas_09_masked
    at1.inputs.output_image = t1_deformed
    at1.inputs.interpolation = 'BSpline'
    at1.inputs.interpolation_parameters = (5,)
    at1.inputs.default_value = 0
    at1.inputs.transforms = [t1_warp, t1_affine]
    at1.inputs.invert_transform_flags = [False, False]
    at1.cmdline
    at1.run()
    # Jacobian of deformation field
    t2=time.time()
    jacobian = CreateJacobianDeterminantImage()
    jacobian.inputs.imageDimension = 3
    jacobian.inputs.deformationField = t1_warp
    jacobian.inputs.outputImage = t1_deformed_jacobian
    jacobian.inputs.num_threads = 6
    jacobian.cmdline
    jacobian.run()
    #run
    try:
        tbuilder.run(plugin='MultiProc', plugin_args={'n_procs':N_PROCESS})
    except(RuntimeError) as err:
        print("RuntimeError:", err)
    else:
        raise
    
if __name__ == '__main__':
    args=get_args()
    STUDY_GRP=args.group; 
    DATA_DIR=args.data;     OUT_DIR=args.output
    SUB_LIST=args.sub_list; N_PROCESS=args.nprocs
    print("The study group is: ", STUDY_GRP)
    print("The input data folder: ", DATA_DIR, type(DATA_DIR))
    print("The output data folder: ", OUT_DIR, type(OUT_DIR) )
    print("The subject list: ",      SUB_LIST, type(SUB_LIST))
    main(STUDY_GRP, DATA_DIR, OUT_DIR, SUB_LIST, N_PROCESS)
