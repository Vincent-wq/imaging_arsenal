# -*- coding: utf-8 -*-
"""
This is the utils lib for imaging preprocessing and processing created by Qing Wang (Vincent)."
2026.1.8
"""
## Libs
# general
import os
import copy
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np
# imaging
import nibabel as nib
from nilearn import image, datasets

def get_custom_atlas_dir(atlas_name:str):
    ## retrive atlas
    atlas_root = "/scratch/atlas/AtlasPack-Vincent/"
    atlas_ver_root = atlas_root+atlas_name
    if atlas_name == "atlas-4S256Parcels":
        atla_label_dir = atlas_ver_root+'/'+atlas_name+'_dseg.tsv'
        atla_img_dir   = atlas_ver_root+'/'+atlas_name+'_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz'
        atlas_coords_dir= atlas_ver_root+'/'+atlas_name+'_coords.npy'
    elif atlas_name == "atlas-HTH":
        atla_label_dir = atlas_ver_root+'/'+'Volumes_names-labels.csv'
        atla_img_dir   = atlas_ver_root+'/'+'MNI152b_atlas_labels_0.5mm.nii.gz'
        atlas_coords_dir= atlas_ver_root+'/'+atlas_name+'_coords.csv'
    elif atlas_name == "atlas-HTH-ARC":
        atla_label_dir = atlas_ver_root+'/'+'Volumes_names-labels.csv'
        atla_img_dir   = atlas_ver_root+'/'+'MNI152b_atlas_labels_0.5mm.nii.gz'
        atlas_coords_dir= atlas_ver_root+'/'+atlas_name+'_coords.csv'
    else:
        print(atlas_name+' not found, please create it using imaging_arsenal!')
        return "", "", ""
    return atla_img_dir, atla_label_dir, atlas_coords_dir

def load_custom_atlas(atlas_name:str):
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
    atla_img_dir, atla_label_dir, atlas_coords_dir = get_custom_atlas_dir(atlas_name)

    atlas_img_path = Path(atla_img_dir)
    labels_path = Path(atla_label_dir)
    coords_path  = Path(atlas_coords_dir)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {atla_label_dir}")

    # Read labels table (auto detect TSV/CSV)
    if labels_path.suffix.lower() == ".tsv":
        lut_df = pd.read_csv(labels_path, sep="\t")
    else:
        lut_df = pd.read_csv(labels_path)

    if atlas_name=="atlas-4S256Parcels":
        # Check for required columns
        expected_cols = {'index', 'label'}
        if not expected_cols.issubset(set(lut_df.columns)):
            raise ValueError(
                f"Labels file must contain columns 'index' and 'label'. "
                f"Found: {list(lut_df.columns)}"
            )
    
        if coords_path.exists():
            coords_ = np.load(coords_path)
        else:
            from nilearn import plotting
            print(" -> Extracting coordinates from NIfTI (One-time setup)...")
            coords_ = plotting.find_parcellation_cut_coords(labels_img=atlas_img_path)
            np.save(coords_path, coords_)

        # Sort by index to match voxel codes
        lut_df = lut_df.sort_values("index").reset_index(drop=True)

        # Build label list (only names)
        labels = lut_df["label"].tolist()

        # Basic description
        description = f"Atlas can be loaded from {atla_img_dir}; labels from {atla_label_dir}"

    elif atlas_name=="atlas-HTH":
        if coords_path.exists():
            coords_ = pd.read_csv(coords_path)
        else:
            print(atlas_name+" coords file missing...")
            return np.nan

        # Sort by index to match voxel codes
        lut_df = lut_df.sort_values("Label").reset_index(drop=True)
        # Build label list (only names)
        labels = lut_df["Label"].tolist()

        # Basic description
        description = f"Atlas can be loaded from {atla_img_dir}; labels from {atla_label_dir}"
    
    elif atlas_name=="atlas-HTH-ARC":
        if coords_path.exists():
            coords_ = pd.read_csv(coords_path)
        else:
            print(atlas_name+" coords file missing...")
            return np.nan
        # Sort by index to match voxel codes
        lut_df = lut_df.sort_values("Label").reset_index(drop=True)
        # Build label list (only names)
        labels = lut_df["Label"].tolist()
        # Basic description
        description = f"Atlas can be loaded from {atla_img_dir}; labels from {atla_label_dir}"

    else:
        print(atlas_name+" not found, please create using imaging arsenal...")
        labels = []
        coords_=[]
        lut_df=[]
        description="Invalide atals, not found."

    # Build a simple atlas object like Nilearn fetchers return
    atlas_obj = SimpleNamespace(
        maps=str(atlas_img_path),
        labels=labels,
        lut=lut_df,
        coords=coords_,
        description=description,
        name=atlas_name
    )

    return atlas_obj

## create the atlas masker
def get_atlas_masker(atlas_name):
    from nilearn import image
    from nilearn.maskers import NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker
    #atlas_img_dir, atlas_label_dir = get_vincent_atlas_dir(atlas_name)
    atlas = load_custom_atlas(atlas_name)
    if os.path.exists(atlas.maps):
        atlas_img = image.load_img(atlas.maps)
        ATLAS_MASKER = NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize=False,
            detrend=False,
            resampling_target="data", # Keep atlas sharp
            strategy="mean",     
            verbose=0
        )
        print(atlas.name, "Atlas loaded...")
        print("Number of labels:", len(atlas.labels))
    else:
        print(f"WARNING: Atlas not found at {atlas.maps}")
        ATLAS_MASKER=None
    return ATLAS_MASKER

## general path scan functions
def get_fmriprep_paths(row, derivatives_path, fmriprep_ver, fix_run_list):
    """
    Constructs paths based on group, session, and specific subject fixes.
    """
    sub_id = str(row['image_id'])
    ses_id = str(row['session'])
    group  = str(row['group'])
    
    # 1. Determine the base directory based on group
    folder_name = f"fmriprep_ses-{ses_id}_{fmriprep_ver}"
    
    base_path = derivatives_path / folder_name / sub_id / f"ses-{ses_id}"
    
    # 2. Handle the "run-01" vs "no-run" naming exception
    # Some subjects missing 'run-01' in their filename
    run_tag = "" if sub_id in fix_run_list else "_run-01"
    
    # 3. Construct File Templates
    paths = {
        't1_file': base_path / 'anat' / f"{sub_id}_ses-{ses_id}_acq-MPRAGE_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
        'mask_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        'bold_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        'confound_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_desc-confounds_timeseries.tsv"
    }
    
    # Check existence and return as string (or empty if missing)
    return {k: str(v) if v.exists() else np.nan for k, v in paths.items()}

##  path scan functions for sz_meditation
def get_fmriprep_paths_sz_meditation(row, derivatives_path, fmriprep_ver, fix_run_list):
    """
    Constructs paths based on group, session, and specific subject fixes.
    """
    sub_id = str(row['image_id'])
    ses_id = str(row['session'])
    group  = str(row['group'])
    
    # 1. Determine the base directory based on group
    folder_name = f"fmriprep_ses-{ses_id}_{fmriprep_ver}"
    
    base_path = derivatives_path / folder_name / sub_id / f"ses-{ses_id}"
    
    # 2. Handle the "run-01" vs "no-run" naming exception
    # Some subjects missing 'run-01' in their filename
    run_tag = "" if sub_id in fix_run_list else "_run-01"
    
    # 3. Construct File Templates
    paths = {
        't1_file': base_path / 'anat' / f"{sub_id}_ses-{ses_id}_acq-MPRAGE{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
        'mask_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        'bold_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        'confound_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_desc-confounds_timeseries.tsv"
    }
    
    # Check existence and return as string (or empty if missing)
    return {k: str(v) if v.exists() else np.nan for k, v in paths.items()}

## ocd scan folder functions (groups are stored in different folders)
def get_ocd_fmriprep_paths(row, derivatives_path, fmriprep_ver, fix_run_list):
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
        'mask_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        'bold_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        'confound_file': base_path / 'func' / f"{sub_id}_ses-{ses_id}_task-rest_acq-mbep2d{run_tag}_desc-confounds_timeseries.tsv"
    }
    
    # Check existence and return as string (or empty if missing)
    return {k: str(v) if v.exists() else "" for k, v in paths.items()}

### main processing function
def process_functional_run(row, conf_params):
    from nilearn.interfaces.fmriprep import load_confounds_strategy, load_confounds
    from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
    from nilearn import image
    """
    Optimized processing for subcortical sensitivity.
    """
    sub_id = row['participant_id']
    ses_id = row['session']
    bold_path = row['bold_file']
    confound_path = row['confound_file']
    output_root = conf_params["OUTPUT_ROOT"]

    ## 
    safe_sub_id = str(sub_id) if str(sub_id).startswith('sub-') else f"sub-{sub_id}"
    image_id = str(row.get('image_id', ''))


    MNI_MASK = conf_params.get('MNI_MASK', None)
    ATLAS_MASKER = conf_params.get('ATLAS_MASKER', None)

    # Output setup
    #out_dir_proc = output_root / f"proc_{conf_params['PROC_TAG']}"
    #out_dir_atlas = output_root / f"proc_{conf_params['PROC_TAG']}" / conf_params['atlas']
    #for p in [out_dir_proc, out_dir_atlas]:
    #    p.mkdir(parents=True, exist_ok=True)

    # 1. Determine GSR States to process based on the single flag
    gsr_setting = conf_params.get('GSR', 'both')
    if gsr_setting == 'both':
        gsr_states = [False, True]
    elif gsr_setting == True:
        gsr_states = [True]
    else:
        gsr_states = [False]
    run_results = []

    try:
        if not os.path.exists(bold_path):
            return [{"sub": sub_id, "ses": ses_id, "status": "FAIL", "reason": "BOLD file missing"}]

        # --- I/O OPTIMIZATION: LOAD AND PREP IMAGES ONLY ONCE ---
        bold_img = image.load_img(bold_path)
        drop_idx = conf_params['DROP_VOLUMES']
        bold_img_cut = image.index_img(bold_img, slice(drop_idx, None))
        
        mask_path = row.get('mask_file', "")
        if os.path.exists(mask_path):
            run_mask = image.load_img(mask_path)
        else:
            run_mask = image.resample_to_img(MNI_MASK, bold_img_cut, interpolation='nearest')

        # --- THE GSR LOOP ---
        for use_gsr in gsr_states:
            gsr_tag = "GSR" if use_gsr else "noGSR"
            
            # Directory Setup for this specific state
            out_dir_proc = output_root / f"proc_{conf_params['PROC_TAG']}_{gsr_tag}"
            out_dir_bold = out_dir_proc / "tpl-MNI152NLin2009cAsym_bold-cleaned"
            out_dir_atlas = out_dir_proc / conf_params['atlas']
            for p in [out_dir_proc, out_dir_bold, out_dir_atlas]:
                p.mkdir(parents=True, exist_ok=True)
            # local strategy
            current_strategy = copy.deepcopy(conf_params['STRATEGY'])
            if use_gsr:
                # 1. Add 'global_signal' to the list of strategies
                if "global_signal" not in current_strategy['strategy']:
                    current_strategy['strategy'].append("global_signal")
                # 2. Add the parameter setting (the 1-parameter standard model)
                current_strategy['global_signal'] = "basic"

            # Load Confounds
            try:
                confounds, sample_mask = load_confounds(bold_path, **current_strategy) # load_confounds_strategy
            except Exception as e:
                run_results.append({"sub": sub_id, "ses": ses_id, "gsr_version": gsr_tag, "status": "FAIL", "reason": f"Confound Error: {e}"})
                continue

            # QC Tracking
            raw_confounds = pd.read_csv(confound_path, sep='\t')
            mean_fd = raw_confounds['framewise_displacement'].mean()
            max_fd = raw_confounds['framewise_displacement'].max()
            
            total_vols = len(raw_confounds)
            vols_after_dummy = total_vols - drop_idx
            
            if sample_mask is not None:
                sample_mask_cut = sample_mask[sample_mask >= drop_idx] - drop_idx
                vols_remaining = len(sample_mask_cut)
            else:
                sample_mask_cut = None
                vols_remaining = vols_after_dummy
            # need fix 
            vols_scrubbed = vols_after_dummy - vols_remaining

            # Slicing Confounds
            if confounds is not None:
                confounds_cut = confounds.iloc[drop_idx:].reset_index(drop=True)

            # Cleaning
            cleaner = NiftiMasker(
                mask_img=run_mask,
                smoothing_fwhm=conf_params['FWHM'], 
                standardize=conf_params['standardize'], 
                standardize_confounds=conf_params['standardize_confounds'], 
                detrend=conf_params['detrend'],        
                low_pass=conf_params['low_pass'],      
                high_pass=conf_params['high_pass'],    
                t_r=conf_params['TR'],
                verbose=0
            )
            cleaned_4d = cleaner.fit_transform(bold_img_cut, confounds=confounds_cut, sample_mask=sample_mask_cut)
            cleaned_img = cleaner.inverse_transform(cleaned_4d)
            
            # Save NIfTI (Disk Saver)
            cleaned_path_str = ""
            if conf_params.get('save_cleaned_nifti', False):
                new_suffix = f"_desc-cleaned{gsr_tag}_bold.nii.gz"
                bold_name = Path(bold_path).name.replace("_desc-preproc_bold.nii.gz", new_suffix).replace(image_id, safe_sub_id)
                cleaned_path = out_dir_bold / bold_name
                cleaned_img.to_filename(str(cleaned_path))
                cleaned_path_str = str(cleaned_path)

            # Extract Atlas
            if ATLAS_MASKER:
                ts_atlas = ATLAS_MASKER.fit_transform(cleaned_img)
                out_name = f"{safe_sub_id}_ses-{ses_id}_{conf_params['atlas']}_desc-{gsr_tag}_timeseries.csv"
                np.savetxt(str(out_dir_atlas / out_name), ts_atlas, delimiter=",")
                shape_res = ts_atlas.shape
            else:
                shape_res = (0,0)
                
            run_results.append({
                "sub": sub_id, 
                "ses": ses_id, 
                "status": "OK", 
                "gsr_version": gsr_tag,
                "mean_fd": mean_fd,
                "max_fd": max_fd,
                "vols_total": total_vols,
                "vols_remaining": vols_remaining,
                "vols_scrubbed": vols_scrubbed,
                "cleaned_path": cleaned_path_str,
                "ts_shape": shape_res
            })

        return run_results

    except Exception as e:
        return [{"sub": sub_id, "ses": ses_id, "status": "FAIL", "reason": str(e)}]


## Processing for local measures

# --- Math Helpers ---
def compute_alff_falff(bold_2d, TR, low_pass=0.08, high_pass=0.01):
    from scipy.fft import rfft, rfftfreq
    """
    Computes ALFF and fALFF.
    Expects bold_2d shape from Nilearn: (n_timepoints, n_voxels)
    """
    # 1. Get the number of timepoints (Axis 0 in Nilearn)
    n_timepoints = bold_2d.shape[0] 
    
    # 2. Transform time domain to frequency domain along the time axis (axis=0)
    freqs = rfftfreq(n_timepoints, d=TR)
    fft_vals = np.abs(rfft(bold_2d, axis=0)) 
    
    # 3. Find indices for the target frequency band
    band_idx = np.where((freqs >= high_pass) & (freqs <= low_pass))[0]
    
    # 4. Calculate ALFF (Sum of amplitudes in the target band across the frequency axis)
    alff = np.sum(fft_vals[band_idx, :], axis=0)
    
    # 5. Calculate fALFF (Target band amplitude / Total amplitude)
    total_amplitude = np.sum(fft_vals, axis=0)
    
    # Avoid division by zero in empty background voxels
    total_amplitude[total_amplitude == 0] = 1 
    falff = alff / total_amplitude
    
    # Returns 1D arrays of shape (n_voxels,) which Nilearn perfectly understands
    return alff, falff

def compute_reho(bold_3d, mask_3d):
    from scipy.stats import rankdata
    reho_map = np.zeros(mask_3d.shape)
    padded_data = np.pad(bold_3d, ((1,1), (1,1), (1,1), (0,0)), mode='constant')
    x, y, z = np.where(mask_3d)
    
    for i in range(len(x)):
        neighborhood = padded_data[x[i]:x[i]+3, y[i]:y[i]+3, z[i]:z[i]+3, :]
        timeseries = neighborhood.reshape(27, -1)
        valid_ts = timeseries[np.any(timeseries, axis=1)]
        
        K, N = valid_ts.shape
        if K < 7: continue
            
        ranks = rankdata(valid_ts, axis=1)
        R_i = np.sum(ranks, axis=0)
        R_mean = np.mean(R_i)
        S = np.sum((R_i - R_mean)**2)
        W = (12 * S) / (K**2 * (N**3 - N))
        reho_map[x[i], y[i], z[i]] = W
    return reho_map

def process_local_metrics(row, conf_params):
    """
    In-memory dual-branch processing for fALFF and ReHo.
    Optimized for high-RAM/multi-core local environments.
    """
    import copy
    import gc
    import nibabel as nib
    from nilearn.interfaces.fmriprep import load_confounds
    from nilearn.maskers import NiftiMasker
    from nilearn import image, masking

    sub_id = row['participant_id']
    ses_id = row['session']
    bold_path = row['bold_file']
    confound_path = row['confound_file']
    output_root = conf_params["OUTPUT_ROOT"]
    
    safe_sub_id = str(sub_id) if str(sub_id).startswith('sub-') else f"sub-{sub_id}"
    MNI_MASK = conf_params.get('MNI_MASK', None)
    ATLAS_MASKER = conf_params.get('ATLAS_MASKER', None)
    
    # Determine GSR states
    gsr_setting = conf_params.get('GSR', 'both')
    gsr_states = [False, True] if gsr_setting == 'both' else ([True] if gsr_setting else [False])
    run_results = []

    try:
        if not os.path.exists(bold_path):
            return [{"sub": sub_id, "status": "FAIL", "reason": "BOLD file missing"}]

        # --- I/O OPTIMIZATION: LOAD ONCE ---
        bold_img = image.load_img(bold_path)
        drop_idx = conf_params['DROP_VOLUMES']
        bold_img_cut = image.index_img(bold_img, slice(drop_idx, None))
        
        mask_path = row.get('mask_file', "")
        if os.path.exists(mask_path):
            run_mask = image.load_img(mask_path)
        else:
            run_mask = image.resample_to_img(MNI_MASK, bold_img_cut, interpolation='nearest')
            
        run_mask_data = run_mask.get_fdata() > 0

        # --- THE GSR LOOP ---
        for use_gsr in gsr_states:
            gsr_tag = "GSR" if use_gsr else "noGSR"
            
            out_dir_proc = output_root / f"proc_Metrics_{conf_params['PROC_TAG']}_{gsr_tag}"
            out_dir_maps = out_dir_proc / "3D_Maps"
            out_dir_atlas = out_dir_proc / "Atlas_Means"
            for p in [out_dir_maps, out_dir_atlas]: p.mkdir(parents=True, exist_ok=True)

            current_strategy = copy.deepcopy(conf_params['STRATEGY'])
            if use_gsr:
                if "global_signal" not in current_strategy['strategy']:
                    current_strategy['strategy'].append("global_signal")
                current_strategy['global_signal'] = "basic"
                
            # CRITICAL: Strip out scrubbing for local metrics
            if "scrub" in current_strategy['strategy']:
                current_strategy['strategy'].remove("scrub")

            # Load Confounds
            try:
                confounds, _ = load_confounds(bold_path, **current_strategy)
            except Exception as e:
                run_results.append({"sub": sub_id, "gsr_version": gsr_tag, "status": "FAIL", "reason": f"Confound Error: {e}"})
                continue
                
            confounds_cut = confounds.iloc[drop_idx:].reset_index(drop=True)

            # -------------------------------------------------------------
            # BRANCH 1: fALFF (Broadband, PSC, Unsmoothed)
            # -------------------------------------------------------------
            cleaner_falff = NiftiMasker(
                mask_img=run_mask,
                smoothing_fwhm=None, 
                standardize="psc", # CRITICAL FOR fALFF
                detrend=False,        
                low_pass=None,     # Broadband
                high_pass=None,    
                t_r=conf_params['TR'],
                verbose=0
            )
            cleaned_2d_falff = cleaner_falff.fit_transform(bold_img_cut, confounds=confounds_cut)
            
            # Compute math
            _, falff_1d = compute_alff_falff(cleaned_2d_falff, TR=conf_params['TR'])
            falff_img = cleaner_falff.inverse_transform(falff_1d)
            
            # Clear memory
            del cleaner_falff, cleaned_2d_falff, falff_1d
            
            # -------------------------------------------------------------
            # BRANCH 2: ReHo (Bandpassed, PSC, Unsmoothed)
            # -------------------------------------------------------------
            cleaner_reho = NiftiMasker(
                mask_img=run_mask,
                smoothing_fwhm=None, 
                standardize="psc", 
                detrend=True,        
                low_pass=0.08,     # Bandpassed
                high_pass=0.01,    
                t_r=conf_params['TR'],
                verbose=0
            )
            cleaned_2d_reho = cleaner_reho.fit_transform(bold_img_cut, confounds=confounds_cut)
            cleaned_3d_reho = cleaner_reho.inverse_transform(cleaned_2d_reho).get_fdata()
            
            # Compute math
            reho_3d = compute_reho(cleaned_3d_reho, run_mask_data)
            reho_img = image.new_img_like(run_mask, reho_3d)
            
            # Clear memory
            del cleaner_reho, cleaned_2d_reho, cleaned_3d_reho
            
            # -------------------------------------------------------------
            # EXTRACTION & SAVING
            # -------------------------------------------------------------
            if ATLAS_MASKER:
                # Extract mean atlas values directly from the 3D maps
                mean_falff = ATLAS_MASKER.fit_transform(falff_img)[0]
                mean_reho = ATLAS_MASKER.fit_transform(reho_img)[0]
                
                # Save the ROI data
                roi_df = pd.DataFrame([mean_falff, mean_reho], index=['fALFF', 'ReHo'])
                roi_out = f"{safe_sub_id}_ses-{ses_id}_desc-{gsr_tag}_LocalMetrics.csv"
                roi_df.to_csv(out_dir_atlas / roi_out)

            # Optional: Save the 3D maps (Small 2MB files)
            if conf_params.get('save_cleaned_nifti', True):
                nib.save(falff_img, out_dir_maps / f"{safe_sub_id}_ses-{ses_id}_desc-{gsr_tag}_falff.nii.gz")
                nib.save(reho_img, out_dir_maps / f"{safe_sub_id}_ses-{ses_id}_desc-{gsr_tag}_reho.nii.gz")
            
            del falff_img, reho_img
            gc.collect() # Force RAM flush

            run_results.append({
                "sub": sub_id, 
                "ses": ses_id, 
                "status": "OK", 
                "gsr_version": gsr_tag
            })

        # Final cleanup for the subject
        del bold_img, bold_img_cut
        gc.collect()
        
        return run_results

    except Exception as e:
        return [{"sub": sub_id, "ses": ses_id, "status": "FAIL", "reason": str(e)}]

## QC related functions
# Simple QC for FD-based scrubbing
def simulate_qc(row, drop_vols=10):
    """Reads only the TSV to calculate motion stats and simulate scrubbing."""

    sub_id = row['participant_id']
    ses_id = row.get('session', 'N/A')
    group = row.get('group', 'Unknown')
    confound_path = row['confound_file']

    # If file is missing, flag it
    import os
    if not os.path.exists(confound_path):
        return {"sub": sub_id, "status": "MISSING"}
    
    try:
        df = pd.read_csv(confound_path, sep='\t')
        # 1. Non-Steady State (Dummy Scan) Detection
        # fMRIPrep detects exact equilibrium points. We take the max of fMRIPrep's 
        # detection or your required minimum to be safe.
        dummy_cols = [c for c in df.columns if c.startswith('non_steady_state_outlier')]
        fmriprep_dummies = len(dummy_cols)
        drop_idx = max(fmriprep_dummies, drop_vols)
        # Slice the dataframe to remove dummy scans
        df_clean = df.iloc[drop_idx:].copy()
        total_usable_vols = len(df_clean)
        if total_usable_vols == 0:
            return {"sub": sub_id, "ses": ses_id, "status": "EMPTY_AFTER_DROP"}
        # 2. Extract Core Vectors (Using .get() safely in case of missing columns)
        fd = df_clean.get('framewise_displacement', pd.Series(np.nan)).values
        dvars = df_clean.get('std_dvars', pd.Series(np.nan)).values
        global_signal = df_clean.get('global_signal', pd.Series(np.nan)).values
        # The first row of derivatives (FD/DVARS) is NaN. We must mask it out.
        valid_mask = ~np.isnan(fd) & ~np.isnan(dvars)
        fd_valid = fd[valid_mask]
        dvars_valid = dvars[valid_mask]
        total_vols_after_dummy = len(df) - drop_vols
        # 3. Calculate Global Signal Stability (Scanner Health)
        # We use the Coefficient of Variation (CV) to normalize across different baseline intensities
        if not np.isnan(global_signal).all() and np.mean(global_signal) != 0:
            gs_cv = np.std(global_signal) / np.mean(global_signal)
        else:
            gs_cv = np.nan
        # 4. The Dual-Threshold Scrubbing Logic    
        strict_fd_mask = (fd_valid > 0.2)
        moderate_fd_mask = (fd_valid > 0.5)
        dual_scrub_mask = (fd_valid > 0.5) & (dvars_valid > 1.5)
        
        vols_scrubbed_strict = np.sum(strict_fd_mask)
        vols_scrubbed_moderate = np.sum(moderate_fd_mask)
        vols_scrubbed_dual = np.sum(dual_scrub_mask)

        return {
            "sub": sub_id,
            "status": "OK",
            "group": row.get('group', 'Unknown'), # Pull clinical group if you have it in df!
            # Motion Distributions
            "mean_fd": np.mean(fd_valid) if len(fd_valid) > 0 else np.nan,
            "max_fd": np.max(fd_valid) if len(fd_valid) > 0 else np.nan,
            "mean_dvars": np.mean(dvars_valid) if len(dvars_valid) > 0 else np.nan,
            "total_usable_vols": total_vols_after_dummy,
            # Hardware/Scanner Health
            "global_signal_cv": gs_cv,
            # Survival Metrics
            "vols_rem_FD02": total_vols_after_dummy - vols_scrubbed_strict,
            "vols_rem_FD05": total_vols_after_dummy - vols_scrubbed_moderate,
            "vols_rem_DUAL": total_vols_after_dummy - vols_scrubbed_dual
        }
        
    except Exception as e:
        return {"sub": sub_id, "status": "ERROR", "error": str(e)}

# Cohort QC function that mimics ABCD/ENIGMA checks
def extract_cohort_qc(row, min_drop_vols=10):
    """
    Extracts multi-dimensional QA metrics from fMRIPrep confound TSVs.
    Mimics large-cohort (ABCD/ENIGMA) dual-threshold and signal variance checks.
    """
    sub_id = row['participant_id']
    ses_id = row.get('session', 'N/A')
    group = row.get('group', 'Unknown')
    confound_path = row['confound_file']
    
    if not isinstance(confound_path, str) or not os.path.exists(confound_path):
        return {"sub": sub_id, "ses": ses_id, "status": "MISSING_FILE"}
    
    try:
        df = pd.read_csv(confound_path, sep='\t')
        
        # 1. Non-Steady State (Dummy Scan) Detection
        # fMRIPrep detects exact equilibrium points. We take the max of fMRIPrep's 
        # detection or your required minimum to be safe.
        dummy_cols = [c for c in df.columns if c.startswith('non_steady_state_outlier')]
        fmriprep_dummies = len(dummy_cols)
        drop_idx = max(fmriprep_dummies, min_drop_vols)
        
        # Slice the dataframe to remove dummy scans
        df_clean = df.iloc[drop_idx:].copy()
        total_usable_vols = len(df_clean)
        
        if total_usable_vols == 0:
            return {"sub": sub_id, "ses": ses_id, "status": "EMPTY_AFTER_DROP"}

        # 2. Extract Core Vectors (Using .get() safely in case of missing columns)
        fd = df_clean.get('framewise_displacement', pd.Series(np.nan)).values
        dvars = df_clean.get('std_dvars', pd.Series(np.nan)).values
        global_signal = df_clean.get('global_signal', pd.Series(np.nan)).values
        
        # The first row of derivatives (FD/DVARS) is NaN. We must mask it out.
        valid_mask = ~np.isnan(fd) & ~np.isnan(dvars)
        fd_valid = fd[valid_mask]
        dvars_valid = dvars[valid_mask]
        
        # 3. Calculate Global Signal Stability (Scanner Health)
        # We use the Coefficient of Variation (CV) to normalize across different baseline intensities
        if not np.isnan(global_signal).all() and np.mean(global_signal) != 0:
            gs_cv = np.std(global_signal) / np.mean(global_signal)
        else:
            gs_cv = np.nan

        # 4. The Dual-Threshold Scrubbing Logic
        # Frame is bad ONLY IF physical motion (FD) AND image intensity (DVARS) both spike
        strict_fd_mask = (fd_valid > 0.2)
        moderate_fd_mask = (fd_valid > 0.5)
        dual_scrub_mask = (fd_valid > 0.5) & (dvars_valid > 1.5)
        
        vols_scrubbed_strict = np.sum(strict_fd_mask)
        vols_scrubbed_moderate = np.sum(moderate_fd_mask)
        vols_scrubbed_dual = np.sum(dual_scrub_mask)

        # 5. Compile the comprehensive QC dictionary
        return {
            "sub": sub_id,
            "ses": ses_id,
            "group": group,
            "status": "OK",
            "fmriprep_dummies": fmriprep_dummies,
            "applied_drop_vols": drop_idx,
            "total_usable_vols": total_usable_vols,
            # Motion Distributions
            "mean_fd": np.mean(fd_valid) if len(fd_valid) > 0 else np.nan,
            "max_fd": np.max(fd_valid) if len(fd_valid) > 0 else np.nan,
            "mean_dvars": np.mean(dvars_valid) if len(dvars_valid) > 0 else np.nan,
            # Hardware/Scanner Health
            "global_signal_cv": gs_cv,
            # Survival Metrics
            "vols_rem_FD02": total_usable_vols - vols_scrubbed_strict,
            "vols_rem_FD05": total_usable_vols - vols_scrubbed_moderate,
            "vols_rem_DUAL": total_usable_vols - vols_scrubbed_dual
        }
        
    except Exception as e:
        return {"sub": sub_id, "ses": ses_id, "status": "ERROR", "error_msg": str(e)}

def evaluate_dual_threshold(qc_df, target_volumes=200):
    """
    Compares the survival rates of FD-Only vs Dual-Threshold scrubbing.
    Calculates exactly how many subjects are 'salvaged' by the advanced method.
    """
    print(f"--- QA Analysis: Standard (FD) vs. Advanced (Dual-Threshold) ---")
    print(f"Target Requirement: Minimum {target_volumes} Usable Volumes\n")

    # 1. Determine who passes under each regime
    qc_df['Pass_FD_Only'] = qc_df['vols_rem_FD05'] >= target_volumes
    qc_df['Pass_DUAL'] = qc_df['vols_rem_DUAL'] >= target_volumes

    # 2. Group by clinical cohort
    summary = qc_df.groupby('group')[['Pass_FD_Only', 'Pass_DUAL']].sum().astype(int)
    
    # 3. Calculate the "Salvage" metrics
    summary['Subjects_Salvaged'] = summary['Pass_DUAL'] - summary['Pass_FD_Only']
    summary['%_Increase'] = ((summary['Subjects_Salvaged'] / summary['Pass_FD_Only']) * 100).round(1)

    # 4. Add a Master Total row
    summary.loc['TOTAL_COHORT'] = summary.sum()
    # Recalculate the percentage for the total row
    total_salvaged = summary.loc['TOTAL_COHORT', 'Subjects_Salvaged']
    total_base = summary.loc['TOTAL_COHORT', 'Pass_FD_Only']
    summary.loc['TOTAL_COHORT', '%_Increase'] = round((total_salvaged / total_base) * 100, 1)

    print(summary.to_string())
    print("\n" + "="*80 + "\n")
    return summary

## mapping atlas from HC
def harmonize_atlas_networks(atlas_df, hc_fc_edges, network_col='network_label_17network', region_col='label', verbose=True):
    from nilearn.connectome import vec_to_sym_matrix
    if verbose:
        print("--- Empirical Network Mapping (Healthy Control Baseline) ---")
        
    harmonized_df = atlas_df.copy()
    n_regions = len(harmonized_df)
    
    # 1. Aggressively scrub bad NaNs
    harmonized_df[network_col] = harmonized_df[network_col].replace(['nan', 'NaN', 'None', '', ' '], np.nan)
    
    # 2. Calculate Baseline FC
    mean_hc_fc = np.mean(hc_fc_edges, axis=0)
    mean_hc_fc_2d = vec_to_sym_matrix(mean_hc_fc, diagonal=np.ones(n_regions))
    
    known_mask = harmonized_df[network_col].notna()
    unknown_mask = harmonized_df[network_col].isna()
    
    known_indices = np.where(known_mask)[0]
    unknown_indices = np.where(unknown_mask)[0]
    
    if verbose:
        print(f" -> Found {len(known_indices)} cortical regions.")
        print(f" -> Found {len(unknown_indices)} subcortical/cerebellar regions to map.")
        
    # 3. Map Unknowns
    known_groups = harmonized_df.loc[known_mask].groupby(network_col).groups
    mapped_labels = []
    
    for u_idx in unknown_indices:
        best_network = None
        max_correlation = -np.inf
        for net_name, net_indices in known_groups.items():
            mean_corr = np.mean(mean_hc_fc_2d[u_idx, net_indices])
            if mean_corr > max_correlation:
                max_correlation = mean_corr
                best_network = net_name
        mapped_labels.append(best_network)
            
    harmonized_df.loc[unknown_mask, network_col] = mapped_labels
    
    # Failsafe Check
    if harmonized_df[network_col].isna().any() or 'nan' in harmonized_df[network_col].values:
        raise ValueError("CRITICAL ERROR: 'nan' values still exist in the network column after mapping.")

    # 4. Generate Properly Sorted Edge Identifiers
    rows, cols = np.tril_indices(n_regions, k=-1)
    
    node_pairs = []
    net_pairs = []
    
    for r, c in zip(rows, cols):
        # Nodes
        node1, node2 = str(harmonized_df[region_col].iloc[r]), str(harmonized_df[region_col].iloc[c])
        node_pairs.append(f"{node1} <-> {node2}")
        
        # Networks (ALPHABETICALLY SORTED to prevent A-B vs B-A splitting)
        net1, net2 = str(harmonized_df[network_col].iloc[r]), str(harmonized_df[network_col].iloc[c])
        sorted_nets = sorted([net1, net2])
        net_pairs.append(f"{sorted_nets[0]} <-> {sorted_nets[1]}")
        
    node_pairs = np.array(node_pairs)
    net_pairs = np.array(net_pairs)
    
    # 5. Generate Group IDs directly inside the function
    unique_groups, group_ids = np.unique(net_pairs, return_inverse=True)
    
    if verbose:
        print(f"✅ Atlas harmonization complete. Generated exactly {len(unique_groups)} unique functional blocks.")
    
    # Return all 4 variables
    return harmonized_df, node_pairs, net_pairs, group_ids

## update harmonize atlas
def harmonize_parallel_cstc_networks(atlas_df, hc_fc_edges, network_col='network_label_17network', region_col='label', verbose=True):
    from nilearn.connectome import vec_to_sym_matrix
    if verbose:
        print("--- Parallel CSTC Network Mapping (Healthy Control Baseline) ---")
        
    harmonized_df = atlas_df.copy()
    n_regions = len(harmonized_df)
    
    # 1. Scrub bad NaNs
    harmonized_df[network_col] = harmonized_df[network_col].replace(['nan', 'NaN', 'None', '', ' '], np.nan)
    
    # 2. Calculate HC Baseline FC
    mean_hc_fc = np.mean(hc_fc_edges, axis=0)
    mean_hc_fc_2d = vec_to_sym_matrix(mean_hc_fc, diagonal=np.ones(n_regions))
    
    known_mask = harmonized_df[network_col].notna()
    unknown_mask = harmonized_df[network_col].isna()
    
    known_indices = np.where(known_mask)[0]
    unknown_indices = np.where(unknown_mask)[0]
    
    if verbose:
        print(f" -> Found {len(known_indices)} cortical regions.")
        print(f" -> Found {len(unknown_indices)} deep regions (Subcortex/Cerebellum) to map.")
        
    # 3. Map Deep Regions to Parallel Domain Networks
    known_groups = harmonized_df.loc[known_mask].groupby(network_col).groups
    mapped_labels = []
    
    for u_idx in unknown_indices:
        region_name = harmonized_df.iloc[u_idx][region_col]
        
        # Determine Structural Domain based on 4S256 Nomenclature
        if 'Cerebellar' in str(region_name):
            domain = 'Cerebellar'
        else:
            domain = 'Subcortex' # Covers LH/RH basal ganglia, thalamus, limbic
            
        best_network = None
        max_correlation = -np.inf
        
        # Find maximal cortical affinity
        for net_name, net_indices in known_groups.items():
            mean_corr = np.mean(mean_hc_fc_2d[u_idx, net_indices])
            if mean_corr > max_correlation:
                max_correlation = mean_corr
                best_network = net_name
                
        # FUSE the Domain and the Network
        parallel_network_name = f"{domain}_{best_network}"
        mapped_labels.append(parallel_network_name)
        
        if verbose and (u_idx % 10 == 0):
            print(f"    Mapped: {str(region_name).ljust(25)} ->  {parallel_network_name} (r = {max_correlation:.3f})")
            
    harmonized_df.loc[unknown_mask, network_col] = mapped_labels

    # 4. Generate Sorted Edge Identifiers
    rows, cols = np.tril_indices(n_regions, k=-1)
    
    node_pairs = []
    net_pairs = []
    
    for r, c in zip(rows, cols):
        node1, node2 = str(harmonized_df[region_col].iloc[r]), str(harmonized_df[region_col].iloc[c])
        node_pairs.append(f"{node1} <-> {node2}")
        
        net1, net2 = str(harmonized_df[network_col].iloc[r]), str(harmonized_df[network_col].iloc[c])
        sorted_nets = sorted([net1, net2])
        net_pairs.append(f"{sorted_nets[0]} <-> {sorted_nets[1]}")
        
    node_pairs = np.array(node_pairs)
    net_pairs = np.array(net_pairs)
    
    # 5. Generate True Group IDs
    unique_groups, group_ids = np.unique(net_pairs, return_inverse=True)
    
    if verbose:
        print(f"\n✅ Parallel Atlas complete. Generated {len(unique_groups)} perfectly segmented CSTC functional blocks.")
    
    return harmonized_df, node_pairs, net_pairs, group_ids
    

def create_stat_map(df, lut, atlas_path, output_path, alpha=0.05, force_recreate=False):
    """
    Loads a high-resolution statistical brain map if it exists. 
    Otherwise, generates it from the dataframe, resamples to 0.5mm, saves it, and returns the image.
    """
    # 1. Check for existing cached map
    if os.path.exists(output_path) and not force_recreate:
        print(f"Loading cached statistical map from: {output_path}")
        return image.load_img(output_path)

    print("Generating new high-resolution statistical map...")
    
    # 2. Build dictionaries
    atlas_mapping = dict(zip(lut['Label'], lut['Abbreviation']))
    stat_values = {}
    
    for index, row in df.iterrows():
        region_str = row['Region']
        p_fdr = row['MIF_P_FDR']
        coef = row['MIF_Coef']
        
        # Only project regions that survive the FDR threshold
        if p_fdr < alpha:
            for abbr in atlas_mapping.values():
                if f"_{abbr}" in region_str: 
                    stat_values[abbr] = coef
                    break

    print(f"Mapped {len(stat_values)} significant regions out of {len(df)} total.")

    # 3. Load and Resample Atlas
    print("Resampling atlas to 0.5mm MNI space (this may take a moment)...")
    original_atlas_img = image.load_img(atlas_path)
    mni_template_dot5mm = datasets.load_mni152_template(resolution=0.5)
    
    highres_atlas_img = image.resample_to_img(
        original_atlas_img, 
        mni_template_dot5mm, 
        interpolation='nearest'
    )
    atlas_data = highres_atlas_img.get_fdata()

    # 4. Project Statistics into Voxel Space
    print("Projecting statistics into voxels...")
    stat_map_data = np.zeros_like(atlas_data)
    
    for label_int, roi_name in atlas_mapping.items():
        if roi_name in stat_values:
            val = stat_values[roi_name]
            stat_map_data[atlas_data == label_int] = val

    # 5. Save and Return
    highres_stat_img = nib.Nifti1Image(stat_map_data, highres_atlas_img.affine, highres_atlas_img.header)
    nib.save(highres_stat_img, output_path)
    print(f"Success! Saved new map to: {output_path}")
    
    return highres_stat_img