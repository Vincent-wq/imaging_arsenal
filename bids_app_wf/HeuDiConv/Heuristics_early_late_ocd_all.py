# -*- coding: utf-8 -*-
"""
Heuristics created by Vincent for PPMI dataset T1/T2/DTI images.
created @ 22th Mar. 2022
merged Ross's heuristics @ 30th Mar. 2022
"""
import os
import logging

lgr = logging.getLogger(__name__)
scaninfo_suffix = '.json'

# scanning protocol details

# converted images: update this part according to the dataset
T1W_SERIES = [
    'MPRAGE'
]

T2W_SERIES = [
    't2_spc_sag_p2_iso'
]
# Added 2026.1.21

DWI_AP_SERIES = ['MB_ep2d_diff-DTI_2.0mmISO_AP',
                 'MB_ep2d_diff-DSI_2.0mmISO_AP'              
]

DWI_PA_SERIES = ['MB_ep2d_diff-DSI_2.0mmISO_PA',
]
# 'MB_ep2d_diff-DTI_2.0mmISO_PA' NO such protocol
# Update 2026.1.19

BOLD_SERIES = [
    'cmrr_mbep2d_bold-rsfMRI-3mmISO'
]

#########################
def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    t1w     = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-MPRAGE_run-{item:02d}_T1w')
    t2w     = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-spcSagIso_run-{item:02d}_T2w')  
    dwi_AP  = create_key('sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-ep2d_dir-AP_run-{item:02d}_dwi') 
    dwi_PA  = create_key('sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-ep2d_dir-PA_run-{item:02d}_dwi') 
    bold    = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_acq-mbep2d_task-rest_run-{item:02d}_bold') 
    
    #swi = create_key('sub-{subject}/{session}/swi/sub-{subject}_run-{item:01d}_swi')
    info = {t1w: [], t2w:[], bold: [], dwi_AP: [], dwi_PA: []}
    
    for idx, s in enumerate(seqinfo):
        # the straightforward scan series
        if s.series_description in T1W_SERIES:# T1
            info[t1w].append(s.series_id)
        elif s.series_description in T2W_SERIES:# T2
            info[t2w].append(s.series_id) 
        elif s.series_description in BOLD_SERIES and s.dim4 > 200:# BOLD
            info[bold].append(s.series_id)     
        elif s.series_description in DWI_AP_SERIES and s.dim4 > 50: # DWI_AP, all derivatives are not included
            info[dwi_AP].append(s.series_id)
        elif s.series_description in DWI_PA_SERIES and s.dim4 > 50: # DWI_PA, all derivatives are not included
            info[dwi_PA].append(s.series_id)
        # if we don't match _anything_ then we want to know!
        else:
            lgr.warning('Skipping unrecognized series description: {}'.format(s.series_description))
    return info
