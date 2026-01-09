#!/usr/bin/env python3
"""
Audit fMRIPrep outputs and generate a per-subject table.

Works with either:
1) A derivatives root (contains dataset_description.json and sub-*/anat, sub-*/func),
   e.g., /bids/derivatives/fmriprep
or
2) An fMRIPrep "output_dir" root that contains a subfolder "fmriprep/".
   In that case, it will auto-detect derivatives under <root>/fmriprep.

Outputs:
- TSV/CSV with one row per subject and columns describing presence/counts of key outputs.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Roots:
    root: Path          # user-provided
    deriv_root: Path    # where sub-*/anat|func live
    report_root: Path   # where sub-*.html live (often deriv_root or deriv_root/fmriprep)
    log_root: Path      # where sub-*/log live (often report_root/sub-*/log)


def _detect_roots(user_root: Path) -> Roots:
    user_root = user_root.resolve()

    # Case A: user_root itself is derivatives root
    if (user_root / "dataset_description.json").exists():
        deriv_root = user_root
        # reports are sometimes in deriv_root, sometimes in deriv_root/fmriprep
        report_root = user_root / "fmriprep" if (user_root / "fmriprep").is_dir() else user_root
        log_root = report_root
        return Roots(user_root, deriv_root, report_root, log_root)

    # Case B: user_root is an "output_dir" that contains fmriprep/ as derivatives
    if (user_root / "fmriprep" / "dataset_description.json").exists():
        deriv_root = user_root / "fmriprep"
        report_root = user_root / "fmriprep"
        log_root = user_root / "fmriprep"
        return Roots(user_root, deriv_root, report_root, log_root)

    raise FileNotFoundError(
        f"Could not detect fMRIPrep derivatives root. "
        f"Expected dataset_description.json in:\n"
        f"  - {user_root}\n"
        f"or\n"
        f"  - {user_root / 'fmriprep'}"
    )

def _list_subjects(deriv_root: Path) -> List[str]:
    subs = []
    for p in sorted(deriv_root.glob("sub-*")):
        if p.is_dir():
            subs.append(p.name.replace("sub-", "", 1))
    return subs


def _exists_any(globs: List[Path]) -> bool:
    return any(g.exists() for g in globs)


def _glob_count(base: Path, pattern: str) -> int:
    return sum(1 for _ in base.glob(pattern))


def _find_one(base: Path, pattern: str) -> Optional[Path]:
    matches = list(base.glob(pattern))
    return matches[0] if matches else None


def _derive_runkey_from_preproc_bold(fname: str) -> str:
    """
    Create a "run key" prefix for matching corresponding confounds/boldref/mask.

    Example:
      sub-01_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    -> sub-01_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym
    """
    # remove _desc-preproc_bold + ext
    runkey = re.sub(r"_desc-preproc_bold(\.nii(\.gz)?)$", "", fname)
    return runkey


def audit_subject(roots: Roots, sub: str) -> Dict:
    deriv_sub = roots.deriv_root / f"sub-{sub}"
    anat_dir = deriv_sub / "anat"
    func_dir = deriv_sub / "func"

    # ---- report + logs (documented locations) ----
    report_html = (roots.report_root / f"sub-{sub}.html")
    log_dir = (roots.log_root / f"sub-{sub}" / "log")

    crashfiles = []
    if log_dir.exists():
        crashfiles = list(log_dir.glob("crash-*.txt")) + list(log_dir.glob("*.pklz"))

    # ---- anatomical key outputs (existence) ----
    anat_preproc_t1w = _find_one(anat_dir, f"sub-{sub}*_desc-preproc_T1w.nii*") if anat_dir.exists() else None
    anat_brainmask  = _find_one(anat_dir, f"sub-{sub}*_desc-brain_mask.nii*") if anat_dir.exists() else None
    anat_aparcaseg  = _find_one(anat_dir, f"sub-{sub}*_desc-aparcaseg_dseg.nii*") if anat_dir.exists() else None
    # common transform naming (varies by version/spaces requested)
    anat_any_xfm_h5  = _find_one(anat_dir, f"sub-{sub}*_mode-image_xfm.h5") if anat_dir.exists() else None

    # surfaces / gifs / gii
    n_surf_gii = _glob_count(anat_dir, f"sub-{sub}*.surf.gii") if anat_dir.exists() else 0

    # ---- functional key outputs (counts + per-run consistency) ----
    n_preproc_bold = 0
    n_confounds_tsv = 0
    n_confounds_json = 0
    n_boldref = 0
    n_func_mask = 0
    n_dtseries = 0

    missing_confounds_for_runs = 0
    missing_boldref_for_runs = 0
    missing_mask_for_runs = 0

    if func_dir.exists():
        preproc_bolds = sorted(func_dir.glob(f"sub-{sub}*_desc-preproc_bold.nii*"))
        n_preproc_bold = len(preproc_bolds)

        conf_tsv = sorted(func_dir.glob(f"sub-{sub}*_desc-confounds_timeseries.tsv"))
        conf_json = sorted(func_dir.glob(f"sub-{sub}*_desc-confounds_timeseries.json"))
        n_confounds_tsv = len(conf_tsv)
        n_confounds_json = len(conf_json)

        boldrefs = sorted(func_dir.glob(f"sub-{sub}*_desc-boldref.nii*"))
        masks = sorted(func_dir.glob(f"sub-{sub}*_desc-brain_mask.nii*"))
        n_boldref = len(boldrefs)
        n_func_mask = len(masks)

        dtseries = sorted(func_dir.glob(f"sub-{sub}*bold.dtseries.nii"))
        n_dtseries = len(dtseries)

        # Per-run matching: for each preproc bold, check corresponding outputs exist
        # We match by the shared prefix runkey + specific suffix.
        conf_tsv_names = {p.name for p in conf_tsv}
        boldref_names = {p.name for p in boldrefs}
        mask_names = {p.name for p in masks}

        for pb in preproc_bolds:
            runkey = _derive_runkey_from_preproc_bold(pb.name)
            expected_conf = f"{runkey}_desc-confounds_timeseries.tsv"
            expected_boldref = f"{runkey}_desc-boldref.nii.gz"
            expected_mask = f"{runkey}_desc-brain_mask.nii.gz"

            if expected_conf not in conf_tsv_names:
                # some versions might output .tsv but with same runkey; this is strict by design
                missing_confounds_for_runs += 1

            # allow nii or nii.gz for boldref/mask
            if not any(n.startswith(f"{runkey}_desc-boldref.nii") for n in boldref_names):
                missing_boldref_for_runs += 1
            if not any(n.startswith(f"{runkey}_desc-brain_mask.nii") for n in mask_names):
                missing_mask_for_runs += 1

    # ---- derive a simple status flag ----
    # Minimal functional "success" heuristic: report + >=1 preproc bold + confounds match all bolds
    ok_min = report_html.exists() and (n_preproc_bold > 0) and (missing_confounds_for_runs == 0)
    status = "OK" if ok_min else "CHECK"

    return {
        "subject": f"sub-{sub}",
        "status": status,

        # reports/logs
        "report_html": report_html.exists(),
        "log_dir": log_dir.exists(),
        "n_crashfiles": len(crashfiles),

        # anat
        "anat_dir": anat_dir.exists(),
        "anat_preproc_T1w": anat_preproc_t1w is not None,
        "anat_brain_mask": anat_brainmask is not None,
        "anat_aparcaseg_dseg": anat_aparcaseg is not None,
        "anat_any_mode-image_xfm_h5": anat_any_xfm_h5 is not None,
        "n_surf_gii": n_surf_gii,

        # func
        "func_dir": func_dir.exists(),
        "n_preproc_bold": n_preproc_bold,
        "n_confounds_tsv": n_confounds_tsv,
        "n_confounds_json": n_confounds_json,
        "n_boldref": n_boldref,
        "n_func_brain_mask": n_func_mask,
        "n_dtseries": n_dtseries,

        # per-run consistency checks
        "missing_confounds_for_preproc_bold": missing_confounds_for_runs,
        "missing_boldref_for_preproc_bold": missing_boldref_for_runs,
        "missing_mask_for_preproc_bold": missing_mask_for_runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fmriprep_root", type=str,
                    help="Path to fMRIPrep derivatives root OR fMRIPrep output_dir containing fmriprep/.")
    ap.add_argument("--out", type=str, default="fmriprep_audit.tsv",
                    help="Output table path (.tsv or .csv). Default: fmriprep_audit.tsv")
    args = ap.parse_args()

    roots = _detect_roots(Path(args.fmriprep_root))
    subs = _list_subjects(roots.deriv_root)

    rows = [audit_subject(roots, s) for s in subs]
    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, sep="\t")

    print(f"Wrote: {out_path}")
    print(df)


if __name__ == "__main__":
    main()
