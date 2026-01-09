# -*- coding: utf-8 -*-
"""
This is the utils lib for imaging preprocessing and processing created by Qing Wang (Vincent)."
2026.1.8
"""
## Libs
# general
import os
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
# imaging
import nibabel as nib

## Atlas related functions
def get_vincent_atlas_dir(atlas_name:str):
    ## retrive atlas
    atlas_root = "/scratch/atlas/AtlasPack-Vincent/"
    atlas_ver_root = atlas_root+atlas_name
    atla_label_dir = atlas_ver_root+'/'+atlas_name+'_dseg.tsv'
    atla_img_dir   = atlas_ver_root+'/'+atlas_name+'_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz'
    return atla_img_dir, atla_label_dir

def load_atlas(atlas_name:str, atlas_img_file: str, labels_file: str):
    """
    Load a 3D label atlas (labels image + labels table) 
    and return a Nilearn‑style atlas object with attributes:
        - maps  : path to atlas image
        - labels: list of label names corresponding to unique values
        - lut   : pandas DataFrame with columns ['index', 'name']
        - description: text description
    
    The labels_file should be a CSV/TSV with at least two columns:
        index (int) and name (str), matching values in atlas_img_file.
    
    Example labels file (TSV):
        index    name
        0        Background
        1        RegionA
        2        RegionB
        ...
    
    Returns:
        SimpleNamespace with attributes: maps, labels, lut, description.
    """

    atlas_img_path = Path(atlas_img_file)
    labels_path = Path(labels_file)

    # Load the atlas and label table
    atlas_img = nib.load(str(atlas_img_path))
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    # Read labels table (auto detect TSV/CSV)
    if labels_path.suffix.lower() == ".tsv":
        lut_df = pd.read_csv(labels_path, sep="\t")
    else:
        lut_df = pd.read_csv(labels_path)

    # Check for required columns
    expected_cols = {'index', 'label'}
    if not expected_cols.issubset(set(lut_df.columns)):
        raise ValueError(
            f"Labels file must contain columns 'index' and 'label'. "
            f"Found: {list(lut_df.columns)}"
        )

    # Sort by index to match voxel codes
    lut_df = lut_df.sort_values("index").reset_index(drop=True)

    # Build label list (only names)
    labels = lut_df["label"].tolist()

    # Basic description
    description = f"Atlas loaded from {atlas_img_file}; labels from {labels_file}"

    # Build a simple atlas object like Nilearn fetchers return
    atlas_obj = SimpleNamespace(
        maps=str(atlas_img_path),
        labels=labels,
        lut=lut_df,
        description=description,
        name=atlas_name
    )

    return atlas_obj

## proc functions
def get_fmriprep_paths(row, derivatives_path, fmriprep_ver, fix_run_list):
    """
    Constructs paths based on group, session, and specific subject fixes.
    """
    sub_id = str(row['image_id'])
    ses_id = str(row['session'])
    group  = str(row['group'])
    
    # 1. Determine the base directory based on group
    if group == 'control':
        folder_name = f"control_fmriprep_ses-{ses_id}_{fmriprep_ver}"
    else:
        folder_name = f"fmriprep_ses-{ses_id}_{fmriprep_ver}"
    
    base_path = derivatives_path / folder_name / sub_id / f"ses-{ses_id}"
    
    # 2. Handle the "run-01" vs "no-run" naming exception
    # Some subjects missing 'run-01' in their filename
    run_tag = "" if sub_id in fix_run_list else "_run-01"
    
    # 3. Construct File Templates
    # Note: Using res-2 as per your fMRIPrep output
    paths = {
        't1_file': base_path / 'anat' / f"{sub_id}_ses-{ses_id}_acq-MPRAGE{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
        #'mask_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        'bold_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        'confound_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_desc-confounds_timeseries.tsv"
    }
    
    # Check existence and return as string (or empty if missing)
    return {k: str(v) if v.exists() else "" for k, v in paths.items()}

### main processing function
def process_functional_run(row, output_root, conf_params):
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
    from nilearn import image
    """
    Optimized processing for subcortical sensitivity.
    """
    sub_id = row['participant_id']
    ses_id = row['session']
    bold_path = row['bold_file']

    MNI_MASK = conf_params['MNI_MASK']
    ATLAS_MASKER = conf_params['ATLAS_MASKER']

    # Output setup
    out_dir_proc = output_root / f"proc_{conf_params['PROC_TAG']}"
    out_dir_4s = output_root / f"proc_{conf_params['PROC_TAG']}" / conf_params['atlas']
    
    for p in [out_dir_proc, out_dir_4s]:
        p.mkdir(parents=True, exist_ok=True)

    try:
        if not os.path.exists(bold_path):
            return {"sub": sub_id, "ses": ses_id, "status": "FAIL", "reason": "File missing"}

        # 1. Load Confounds (With CompCor strategy)
        try:
            confounds, sample_mask = load_confounds_strategy(
                bold_path,
                **conf_params['STRATEGY']
            )
        except Exception as e:
             return {"sub": sub_id, "ses": ses_id, "status": "FAIL", "reason": f"Confound Error: {e}"}

        bold_img = image.load_img(bold_path)

        # 2. Slicing (Dummy Scans)
        drop_idx = conf_params['DROP_VOLUMES']
        bold_img_cut = image.index_img(bold_img, slice(drop_idx, None))
        
        if confounds is not None:
            confounds_cut = confounds.iloc[drop_idx:].reset_index(drop=True)
            if sample_mask is not None:
                # Adjust sample_mask indices
                sample_mask_cut = sample_mask[sample_mask >= drop_idx] - drop_idx
            else:
                sample_mask_cut = None
        
        # 3. Generate Mask (Prefer fMRIPrep mask)
        # Using the standard MNI mask here, but you should check if 'bold_mask_file' exists in your df
        run_mask = image.resample_to_img(MNI_MASK, bold_img_cut, interpolation='nearest')

        # 4. Cleaning & Denoising
        # smoothing (FWHM) applied here
        cleaner = NiftiMasker(
            mask_img=run_mask,
            smoothing_fwhm=conf_params['FWHM'], 
            standardize=True,  
            detrend=conf_params['detrend'],        #
            low_pass=conf_params['low_pass'],      # EXPLICITLY NONE. Skip low-pass filtering.
            high_pass=conf_params['high_pass'],    # EXPLICITLY NONE. Drift is handled by the confounds.
            t_r=conf_params['TR'],
            verbose=0
        )
        
        cleaned_4d = cleaner.fit_transform(
            bold_img_cut, 
            confounds=confounds_cut, 
            sample_mask=sample_mask_cut
        )
        
        # Revert to image
        cleaned_img = cleaner.inverse_transform(cleaned_4d)
        
        # Save Cleaned Image (Optional: This file will be large)
        # Only save if you need to visualize the cleaned data later
        bold_name = Path(bold_path).name.replace("_desc-preproc_bold.nii.gz", "_desc-cleaned_bold.nii.gz").replace(f"{row['image_id']}", f"sub-{sub_id}")
        cleaned_path = out_dir_proc / bold_name
        cleaned_img.to_filename(str(cleaned_path))

        # 5. Extract 4S Atlas Signal
        if ATLAS_MASKER:
            ts_4s = ATLAS_MASKER.fit_transform(cleaned_img)
            # Save Time Series
            out_name = f"sub-{sub_id}_ses-{ses_id}_{conf_params['atlas']}_timeseries.csv"
            np.savetxt(str(out_dir_4s / out_name), ts_4s, delimiter=",")
            shape_res = ts_4s.shape
        else:
            shape_res = (0,0)
        return {
            "sub": sub_id, 
            "ses": ses_id, 
            "status": "OK", 
            "cleaned_path": str(cleaned_path),
            "ts_shape": shape_res
        }

    except Exception as e:
        return {"sub": sub_id, "ses": ses_id, "status": "FAIL", "reason": str(e)}