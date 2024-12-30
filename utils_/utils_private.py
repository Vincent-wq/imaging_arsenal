# -*- coding: utf-8 -*-
"""This is the utils library created by Qing Wang (Vincent), it may be extended to a new lib in the future, 
but by now, the only purpose of this lib is to support his projects."

"""
import pandas as pd
import numpy as np
#
import pickle
from pathlib import Path
#
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, norm
#### project data related

# merge class info
def participant_2_class(dict_, participant_id_):
    # convert group using dict
    return str(dict_[participant_id_])

def group_class_merge(group_str, class_str):
    # for sz-meditation project
    if group_str == 'meditation' and class_str=='1':
        return 'responder'
    elif group_str == 'meditation' and class_str=='2':
        return 'non-responder'
    elif group_str == 'control' and class_str=='1':
        return 'lucky_guy'
    elif group_str == 'control' and class_str!='1':
        # This is created for sub-JZS0086 in control group for missing data
        return 'control'
    else:
        return ''
        
#### System tool
def report_cpu():
    # report system info
    import os
    import psutil
    num_cpus = os.cpu_count()
    print(f"Total CPUs: {num_cpus}")

    # Get the total memory
    total_memory = psutil.virtual_memory().total
    available_memory = psutil.virtual_memory().available
    # Convert to GB
    total_memory_gb = total_memory / (1024 ** 3)
    available_memory_gb = available_memory / (1024 ** 3)
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Available Memory: {available_memory_gb:.2f} GB")

def read_pkl(file_name_):
    ## read python pkl data into dict
    import pickle
    import os
    if os.path.exists(file_name_):
        with open(file_name_, 'rb') as file:
            data = pickle.load(file)
    else:
        data=np.nan
        print("file does not exist!")
    return data

def save_pkl(any_data_, file_name_):
    import pickle
    f = open(str(file_name_), 'wb')
    pickle.dump(any_data_, f)
    f.close()
    return 1

#### File IO
## pickle IO
def sv_pickle(obj_, file_name):
    # Save the fitted GroupSparseCovarianceCV object to a file
    with open(file_name, 'wb') as f:
        pickle.dump(obj_, f)
        
def read_pickle(file_):
    # Load the GroupSparseCovarianceCV object from the file
    with open(file_, 'rb') as f:
        res_obj = pickle.load(f)
    return res_obj

#### data science
def report_nan_array(arr_):
    # report index of nan elements in np.array
    nan_indices = np.where(np.isnan(arr_))
    # Combine row and column indices
    nan_coordinates = list(zip(nan_indices[0], nan_indices[1]))
    print("Coordinates of NaN values:", nan_coordinates)
    return nan_coordinates

def describe_array(arr):
    import numpy as np
    import scipy.stats as stats
    """
    Report descriptive statistics of a numpy array.

    Parameters:
    arr (numpy.ndarray): The array to describe.

    Returns:
    dict: A dictionary containing descriptive statistics.
    """
    desc_stats = {
        'count': np.size(arr),
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1),  # Sample standard deviation (ddof=1)
        'min': np.min(arr),
        '25%': np.percentile(arr, 25),
        'median': np.median(arr),
        '75%': np.percentile(arr, 75),
        'max': np.max(arr),
        'skewness': stats.skew(arr),
        'kurtosis': stats.kurtosis(arr)
    }
    return desc_stats

def report_min_max_array(arr_, N_):
    from collections import Counter
    res_dict = {}
    # Step 1: Compute the absolute values of the array
    abs_arr = np.abs(arr_)
    # Step 2: Find the indices of the max and min absolute values
    flat_indices = np.argpartition(-abs_arr.flatten(), N_-1)[:N_]  # `-abs_arr` to sort in descending order
    nd_indices = np.unravel_index(flat_indices, arr_.shape)
    top_values = arr_[nd_indices]
    res_dict['values'] = top_values
    res_dict['idx'] = list(zip(*nd_indices))
    # Print the results
    print(f"The {N_} largest absolute values are: {res_dict['values']}")
    print(f"Their corresponding indices are: {res_dict['idx']}")
    # count index freq
    flattened_list = [item for tup in res_dict['idx'] for item in tup]
    element_counts = Counter(flattened_list)
    print(element_counts)
    return res_dict

def fisherz_corr(df_, group_dict, feature_list):
    """
    Compares the correlation coefficients of two features between two groups using Fisher's Z-test.
    
    Parameters:
    df_: dataframe.
    group_dict: dictionary for group labels, like {'treatment':'late_ocd', 'control':'early_ocd'}.
    feature_list: two features to calculate the correlation for two groups.
    
    Returns:
    z_stat: Z-statistic for the comparison.
    p_value: P-value for the comparison.
    r1: correlation for the treatment group.
    r2: correlation for the control group.
    """
    ### remove nan
    df_tmp = df_[["group"]+feature_list].dropna()

    ### get group data
    treatment_g_label = group_dict['treatment']
    control_g_label = group_dict['control']
    treatment_f1 = np.asarray(df_tmp[df_tmp['group']==treatment_g_label][feature_list[0]])
    treatment_f2 = np.asarray(df_tmp[df_tmp['group']==treatment_g_label][feature_list[1]])
    control_f1   = np.asarray(df_tmp[df_tmp['group']==control_g_label][feature_list[0]])
    control_f2   = np.asarray(df_tmp[df_tmp['group']==control_g_label][feature_list[1]])

    # Calculate correlation coefficients for each group
    r1, _ = pearsonr(treatment_f1, treatment_f2)
    r2, _ = pearsonr(control_f1, control_f2)
    
    # Fisher's Z-transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
     
    # Sample sizes
    n1 = len(treatment_f1)
    n2 = len(control_f1)
    
    # Standard error
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    
    # Z-statistic
    z_stat = (z1 - z2) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value, r1, r2

def partial_corr_multi(x, y, covariates):
    """
    Calculate partial correlation between x and y controlling for multiple covariates.
    
    Parameters:
    - x: Array of values for variable X
    - y: Array of values for variable Y
    - covariates: 2D array where each column is a covariate
    
    Returns:
    - partial correlation coefficient
    """
    # Regress x on covariates
    model_x = LinearRegression().fit(covariates, x)
    res_x = x - model_x.predict(covariates)
    
    # Regress y on covariates
    model_y = LinearRegression().fit(covariates, y)
    res_y = y - model_y.predict(covariates)
    
    # Compute the correlation of residuals
    r, _ = pearsonr(res_x, res_y)
    return r

def compare_partial_corr_multi(df_, group_dict, feature_list, cov_list):
    """
    Compares the partial correlation coefficients of two features between two groups using Fisher's Z-test.
    
    Parameters:
    df_: dataframe.
    group_dict: dictionary for group labels.
    feature_list: two features to calculate the correlation for two groups.
    cov_list: covariate to control when calculating partical correlation.

    Returns:
    z_stat: Z-statistic for the comparison.
    p_value: P-value for the comparison.
    r1: correlation for the treatment group.
    r2: correlation for the control group.
    """
    ###
    ### remove nan
    df_tmp = df_[["group"]+feature_list+cov_list].dropna()

    ### get group data
    treatment_g_label = group_dict['treatment']
    control_g_label = group_dict['control']
    treatment_f1 = np.asarray(df_tmp[df_tmp['group']==treatment_g_label][feature_list[0]])
    treatment_f2 = np.asarray(df_tmp[df_tmp['group']==treatment_g_label][feature_list[1]])
    treatment_cov = np.asarray(df_tmp[df_tmp['group']==treatment_g_label][cov_list])

    control_f1   = np.asarray(df_tmp[df_tmp['group']==control_g_label][feature_list[0]])
    control_f2   = np.asarray(df_tmp[df_tmp['group']==control_g_label][feature_list[1]])
    control_cov  = np.asarray(df_tmp[df_tmp['group']==control_g_label][cov_list])

    # Calculate partial correlations
    r1 = partial_corr_multi(treatment_f1, treatment_f2, treatment_cov)
    r2 = partial_corr_multi(control_f1, control_f2, control_cov)

    # Fisher's Z-transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Sample sizes
    n1 = len(treatment_f1)
    n2 = len(control_f1)
    k = treatment_cov.shape[1]  # Number of covariates

    # Standard error
    se = np.sqrt(1 / (n1 - k - 3) + 1 / (n2 - k - 3))

    # Z-statistic
    z_stat = (z1 - z2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value, r1, r2


#### Image related
def insert_string(old_str, insert_str, pos):
    return old_str[:pos] + insert_str + old_str[pos:]

def fs_2_df(subj_df, fs_folder, session_id, col_sel_list, save_path):
    if session_id == '':
        session_id = str(fs_folder).split('/')[-1].split('_')[1]

    files_2_read={'seg'      : ['aseg_stats.txt', 'wmparc_stats.txt'],
                  'Destrieux': {'ct': '.a2009s.thickness.txt',  'area':'.a2009s.area.txt',   'volume':'.a2009s.volume.txt'},
                  'DKT'      : {'ct': '.DKTatlas.thickness.txt','area':'.DKTatlas.area.txt', 'volume':'.DKTatlas.volume.txt'}}
    print('Reading freesurfer stats in folder ' , str(fs_folder), ' ...')
    # preparing files
    subcortical_file = fs_folder / (files_2_read['seg'][0]); wm_file = fs_folder / (files_2_read['seg'][1]);
    # Des parcellation
    lh_Des_ct_file = fs_folder / ('lh'+files_2_read['Destrieux']['ct']);      rh_Des_ct_file = fs_folder /  ('rh'+files_2_read['Destrieux']['ct'])
    lh_Des_vol_file = fs_folder / ('lh'+files_2_read['Destrieux']['volume']); rh_Des_vol_file = fs_folder / ('rh'+files_2_read['Destrieux']['volume'])
    lh_Des_area_file = fs_folder / ('lh'+files_2_read['Destrieux']['area']);  rh_Des_area_file = fs_folder / ('rh'+files_2_read['Destrieux']['area'])
    # DKT parcellation
    lh_DKT_area_file = fs_folder / ('lh'+files_2_read['DKT']['area']);  rh_DKT_area_file = fs_folder / ('rh'+files_2_read['DKT']['area'])
    lh_DKT_ct_file = fs_folder / ('lh'+files_2_read['DKT']['ct']);      rh_DKT_ct_file = fs_folder /  ('rh'+files_2_read['DKT']['ct'])
    lh_DKT_vol_file = fs_folder / ('lh'+files_2_read['DKT']['volume']); rh_DKT_vol_file = fs_folder / ('rh'+files_2_read['DKT']['volume'])

    # drop_list
    aseg_drop = ["EstimatedTotalIntraCranialVol"]; 
    wm_drop   = ["MaskVol", "EstimatedTotalIntraCranialVol", "CerebralWhiteMatterVol", "rhCerebralWhiteMatterVol", "lhCerebralWhiteMatterVol"]
    
    subj_df.loc[:, 'session'] = len(subj_df)*[session_id]

    # read subcortical file
    subcortical_tab = pd.read_csv(subcortical_file, sep='\t', header=0).rename(columns={'Measure:volume':'participant_id'})
    cols_aseg_dict = {'3rd-Ventricle':'Ventricle_3rd', '4th-Ventricle':'Ventricle_4th', '5th-Ventricle':'Ventricle_5th'}
    subcortical_tab.rename(columns=cols_aseg_dict, inplace=True)
    subcortical_tab['eTIV']=subcortical_tab['EstimatedTotalIntraCranialVol']
    subcortical_tab.drop(aseg_drop, axis=1, inplace=True)

    # read wm_file
    res = pd.merge(subj_df, subcortical_tab, on='participant_id')
    wm_tab = pd.read_csv(wm_file, sep='\t', header=0).rename(columns={'Measure:volume':'participant_id'}); wm_tab.drop(wm_drop, axis=1, inplace=True)
    res1 = pd.merge(res, wm_tab, on='participant_id')
    
    # read Des/DKT parcelation data
    lh_Des_ct_tab  = pd.read_csv(lh_Des_ct_file,  sep='\t', header=0).rename(columns={insert_string(str(lh_Des_ct_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 
    rh_Des_ct_tab  = pd.read_csv(rh_Des_ct_file,  sep='\t', header=0).rename(columns={insert_string(str(rh_Des_ct_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 
    lh_Des_vol_tab = pd.read_csv(lh_Des_vol_file, sep='\t', header=0).rename(columns={insert_string(str(lh_Des_vol_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'});  
    rh_Des_vol_tab = pd.read_csv(rh_Des_vol_file, sep='\t', header=0).rename(columns={insert_string(str(rh_Des_vol_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 
    lh_Des_area_tab = pd.read_csv(lh_Des_area_file, sep='\t', header=0).rename(columns={insert_string(str(lh_Des_area_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 
    rh_Des_area_tab = pd.read_csv(rh_Des_area_file, sep='\t', header=0).rename(columns={insert_string(str(rh_Des_area_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 
    
    # DKT atlas
    lh_DKT_ct_tab  = pd.read_csv(lh_DKT_ct_file,  sep='\t', header=0).rename(columns={insert_string(str(lh_DKT_ct_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'});  
    rh_DKT_ct_tab  = pd.read_csv(rh_DKT_ct_file,  sep='\t', header=0).rename(columns={insert_string(str(rh_DKT_ct_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'});   
    lh_DKT_vol_tab = pd.read_csv(lh_DKT_vol_file, sep='\t', header=0).rename(columns={insert_string(str(lh_DKT_vol_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 
    rh_DKT_vol_tab = pd.read_csv(rh_DKT_vol_file, sep='\t', header=0).rename(columns={insert_string(str(rh_DKT_vol_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'});  
    lh_DKT_area_tab = pd.read_csv(lh_DKT_area_file, sep='\t', header=0).rename(columns={insert_string(str(lh_DKT_area_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 
    rh_DKT_area_tab = pd.read_csv(rh_DKT_area_file, sep='\t', header=0).rename(columns={insert_string(str(rh_DKT_area_file).split('/')[-1][:-4], '.aparc', 2):'participant_id'}); 

    # 
    seg_Des_tab = pd.merge(res1, lh_Des_ct_tab, on='participant_id');          seg_Des_tab = pd.merge(seg_Des_tab, rh_Des_ct_tab, on='participant_id'); 
    seg_Des_tab = pd.merge(seg_Des_tab, lh_Des_vol_tab, on='participant_id');  seg_Des_tab = pd.merge(seg_Des_tab, rh_Des_vol_tab, on='participant_id'); 
    seg_Des_tab = pd.merge(seg_Des_tab, lh_Des_area_tab, on='participant_id'); seg_Des_tab = pd.merge(seg_Des_tab, rh_Des_area_tab, on='participant_id'); 

    seg_DKT_tab = pd.merge(res1, lh_DKT_ct_tab, on='participant_id');          seg_DKT_tab = pd.merge(seg_DKT_tab, rh_DKT_ct_tab, on='participant_id');  
    seg_DKT_tab = pd.merge(seg_DKT_tab, lh_DKT_vol_tab, on='participant_id');  seg_DKT_tab = pd.merge(seg_DKT_tab, rh_DKT_vol_tab, on='participant_id'); 
    seg_DKT_tab = pd.merge(seg_DKT_tab, lh_DKT_area_tab, on='participant_id'); seg_DKT_tab = pd.merge(seg_DKT_tab, rh_DKT_area_tab, on='participant_id');   

    # return data
    all_data = {'Des': seg_Des_tab, 'DKT': seg_DKT_tab}
    # correct column names and replace '-' with '_'
    for k, v in all_data.items():
        #v.index   = [x.replace('-','_') for x in v.index]
        v.columns = [x.replace('-','_') for x in v.columns]
    
    print('In session '+session_id+" : Des atlas: ", len(seg_Des_tab), ", DKT atlas: ",len(seg_DKT_tab))

    if save_path !='':
        print("saving...")
        seg_Des_tab.to_csv(save_path+'/Des_ses-'+session_id+'.csv') 
        seg_DKT_tab.to_csv(save_path+'/DKT_ses-'+session_id+'.csv')
        if len(col_sel_list)!=0:
            seg_Des_tab.loc[:,col_sel_list].to_csv(save_path+'/Des_ses-'+session_id+'_sel.csv')
            seg_DKT_tab.loc[:,col_sel_list].to_csv(save_path+'/DKT_ses-'+session_id+'_sel.csv')
    else:
        print("data not saved...")
    return all_data

## functional imagine related
def plot_all_ICs(res_dict, curr_method='canICA', curr_ses=0, order_list=[]):
    import matplotlib.pyplot as plt
    import nilearn.image as image
    from nilearn.plotting import plot_prob_atlas, plot_stat_map, show
    
    ic_list = []; 
    ic_mask_list = []

    # Plot 20 subplots in one figure
    fig, axes = plt.subplots(4, 5, figsize=(24, 13.5))
    if isinstance(res_dict, dict):
        ic_images_ = res_dict[curr_method][curr_ses]['image']
    elif isinstance(res_dict, str):
        ic_images_ = image.load_img(res_dict)
    else:
        ic_images_ = res_dict

    if len(order_list) != 0:
        for ((i, cur_img), cur_ax) in zip(enumerate(image.iter_img(ic_images_)), axes.flatten()):
            j = order_list[i]
            cur_img = image.index_img(ic_images_, j)
            ic_list.append(cur_img)
            ic_mask_list.append(image.math_img('img != 0', img=cur_img))
            plot_stat_map(
                cur_img,
                display_mode="z",
                title=f"Component {int(j)}",
                cut_coords=1,
                colorbar=True,
                #cmap = cmap_trans,
                axes=cur_ax
                )
    else:
        for ((i, cur_img), cur_ax) in zip(enumerate(image.iter_img(ic_images_)), axes.flatten()):
            ic_list.append(cur_img)
            ic_mask_list.append(image.math_img('img != 0', img=cur_img))
            plot_stat_map(
                cur_img,
                display_mode="z",
                title=f"Component {int(i)}",
                cut_coords=1,
                colorbar=True,
                #cmap = cmap_trans,
                axes=cur_ax
                )
    fig.suptitle(curr_method + ' at session '+str(curr_ses), x=0.5, y=0.925, fontsize=24)
    plt.show()
    return ic_list, ic_mask_list

def precision2pcorr(pre_mat, fill_diag=1):
    pcorr_mat = - pre_mat / np.sqrt(np.outer(np.diag(pre_mat), np.diag(pre_mat)))
    #print(pcorr_mat)
    if fill_diag!="":
        np.fill_diagonal(pcorr_mat, float(fill_diag))
    return pcorr_mat

## compute FC
def get_FC(subject_time_series, N_JOBS = 32):
    from nilearn.connectome import GroupSparseCovarianceCV
    from sklearn.covariance import GraphicalLassoCV
    from nilearn.connectome import ConnectivityMeasure
    #
    correlation_measure = ConnectivityMeasure(kind="correlation")
    corr = correlation_measure.fit_transform(subject_time_series)
    #
    gsc = GroupSparseCovarianceCV(verbose=2, n_jobs=N_JOBS)
    gsc.fit(subject_time_series)
    #
    gl = GraphicalLassoCV(verbose=2, n_jobs=N_JOBS)
    gl.fit(np.concatenate(subject_time_series))
    return corr, gsc, gl

## FC comparison
def comp_fc(group1_fcs, group2_fc2, lables_, P_TH):
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    t_values, p_values = ttest_ind(group1_fcs, group2_fc2, axis=2)
    ## ignore nan values
    flat_p = p_values.flatten()
    valid_indices = ~np.isnan(flat_p)
    valid_p_values = flat_p[valid_indices]

    # Correct p-values for multiple comparisons using FDR
    _, corrected_p_values, _, _ = multipletests(valid_p_values, method="fdr_bh")

    corrected_p_matrix = np.full(flat_p.shape, np.nan) 
    corrected_p_matrix[valid_indices] = corrected_p_values

    corrected_p_matrix = corrected_p_matrix.reshape(p_values.shape)
    #print(corrected_p_matrix)
    #
    significant_uncorrected = p_values < P_TH
    significant_corrected = corrected_p_matrix < P_TH
    # 
    n_regions = group1_fcs.shape[0]
    edge_indices = [(i, j) for i in range(n_regions) for j in range(i + 1, n_regions)]  # Upper triangle indices
    results = []
    for i, j in edge_indices:
        if corrected_p_matrix[i, j] <= P_TH or p_values[i, j] <= P_TH: # 
            results.append({
                "Region 1 label": i,
                "Region 2 label": j,
                "Region 1": lables_[i],
                "Region 2": lables_[j],
                "T-Value": t_values[i, j],
                "P-Value (Uncorrected)": p_values[i, j],
                "P-Value (Corrected)": corrected_p_matrix[i, j],
                "Significant (Uncorrected)": significant_uncorrected[i, j],
                "Significant (Corrected)": significant_corrected[i, j]
        })
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

####
def report_stats(df_in, cols_, group_dict, session_dict, th_p):
    ## doing basic statistical tests and output dataframe
    col_list = ['covariate', 'session', 'n_treatment', 'n_total_treatment', 'p_treatment',
                'n_control',  'n_total_control', 'p_control', 
                "n_sample", 'n_total', "n_missing", "p_missing",
                "test", "cat_levels", "statistic", "p_val", "significant",
                "mean_t", 'std_t', 'mean_c','std_c',
                'ct_index', 'ct_column', 'cross_tab', 'p_cross_tab',
                ]
    group_col = list(group_dict.keys())[0]
    group_labels = list(group_dict.values())[0]
    
    res_df = pd.DataFrame(columns=col_list)

    if len(session_dict) >= 1:
        print('Sessional data summary...')
        time_col = list(session_dict.keys())[0]
        time_labels = list(session_dict.values())[0]
        for time_label_ in time_labels:
            print("processing session -> ",str(time_label_))
            session_dict_ = {"session": time_label_}
            df_raw = df_in[df_in[time_col]==time_label_].copy()
            res_df = report_stats_cols(df_raw, res_df, group_col, group_labels, session_dict_, cols_, th_p)
    else:
        print('Baseline data summary...')
        res_df = report_stats_cols(df_in, res_df, group_col, group_labels, {}, cols_, th_p)
    
    return res_df

#### data related
def report_stats_cols(df_raw, res_df, group_col, group_labels, session_dict, cols_, th_p):
    from scipy.stats import chisquare, chi2_contingency, fisher_exact
    from scipy.stats.contingency import association
    import statsmodels.stats.weightstats as ws
    n_rows=len(res_df)
    for  _i in range(len(cols_)):
        print(cols_[_i], 'processing...')
        # basic info
        res_df.loc[n_rows+_i, "covariate"]  = cols_[_i]
        # get raw data
        df_ = df_raw.loc[:, [group_col, cols_[_i]]].copy()
        # convert col data to numeiric else nan.
        df_[cols_[_i]] = pd.to_numeric(df_[cols_[_i]], errors='coerce')
        # decide whether binary
        _col_levels = df_.loc[:, cols_[_i]].unique()
        res_df.loc[n_rows+_i, "cat_levels"]  = len(_col_levels)
        # Get raw data for 2 groups
        df_1_raw = df_[df_[group_col]==group_labels[0]][cols_[_i]]
        df_2_raw = df_[df_[group_col]==group_labels[1]][cols_[_i]]
        # Get total n
        res_df.loc[n_rows+_i, "n_total_treatment"] = len(df_1_raw)
        res_df.loc[n_rows+_i, "n_total_control"]   = len(df_2_raw)
        res_df.loc[n_rows+_i, "n_total"]   = len(df_)
        # Remove na
        df_1 = df_1_raw[~np.isnan(df_1_raw)]
        df_2 = df_2_raw[~np.isnan(df_2_raw)]
        # Get real n
        res_df.loc[n_rows+_i, "n_treatment"] = len(df_1)
        res_df.loc[n_rows+_i, "n_control"]   = len(df_2)
        res_df.loc[n_rows+_i, "n_sample"]    = len(df_[cols_[_i]][~np.isnan(df_[cols_[_i]])])
        res_df.loc[n_rows+_i, "n_missing"]   = res_df.loc[n_rows+_i, "n_total"] - res_df.loc[n_rows+_i, "n_sample"] 
        # Cal percentage
        res_df.loc[n_rows+_i, "p_treatment"] = (res_df.loc[n_rows+_i, "n_treatment"] / res_df.loc[n_rows+_i, "n_total_treatment"]) * 100
        res_df.loc[n_rows+_i, "p_control"]   = (res_df.loc[n_rows+_i, "n_control"] / res_df.loc[n_rows+_i, "n_total_control"]) * 100
        res_df.loc[n_rows+_i, "p_missing"]   = (res_df.loc[n_rows+_i, "n_missing"] / res_df.loc[n_rows+_i, "n_total"]) * 100
        if res_df.loc[n_rows+_i, "cat_levels"] >0 and res_df.loc[n_rows+_i, "cat_levels"] < 7:
            tab_= pd.crosstab(df_.loc[:, group_col], df_.loc[:,cols_[_i]], margins=False)
            #print(tab_)
            total = tab_.sum().sum()
            tab_per = tab_.apply(lambda x: x/total)
            # test
            stat, p, _no_, no_ = chi2_contingency(tab_)
            res_df.loc[n_rows+_i, "test"]   = "Pearson Chi-square"
            res_df.loc[n_rows+_i, "statistic"]   = stat
            res_df.loc[n_rows+_i, "p_val"]       = p
            #print(len(np.asarray(tab_.index)))
            res_df.at[n_rows+_i, "ct_index"]    = list(tab_.index)
            res_df.at[n_rows+_i, "ct_column"]   = list(tab_.columns)
            res_df.at[n_rows+_i, "cross_tab"]   = np.asarray(tab_)
            res_df.at[n_rows+_i, "p_cross_tab"] = np.asarray(tab_per)
            #print(tab_)
            if p < th_p:
                res_df.loc[n_rows+_i, "significant"] = 1
                print(cols_[_i], 'is a categorical varilbes with '+str(len(_col_levels)), 'levels, ', 'Pearson Chi-square test:', stat ,p)
                #print('Variable:', cols_[_i], ', Pearson Chi-square test:', stat ,p)
            else:
                res_df.loc[n_rows+_i, "significant"] = 0
        elif res_df.loc[n_rows+_i, "cat_levels"] >= 7:
            res_df.loc[n_rows+_i, "mean_t"] = df_1.mean()
            res_df.loc[n_rows+_i, "std_t"]  = df_1.std()
            res_df.loc[n_rows+_i, "mean_c"] = df_2.mean()
            res_df.loc[n_rows+_i, "std_c"]  = df_2.std()
            # t-test
            t_stat, t_pval, t_df = ws.ttest_ind(df_1, df_2, alternative='two-sided', usevar='pooled')
            res_df.loc[n_rows+_i, "test"] = 't-test'
            res_df.loc[n_rows+_i, "statistic"] = t_stat
            res_df.loc[n_rows+_i, "p_val"]     = t_pval
            if t_pval < th_p:
                res_df.loc[n_rows+_i, "significant"] = 1
                print(cols_[_i],'is a continious varilbes, ', 'tstat =%.6f, pvalue = %.6f, df = %i'%(t_stat, t_pval, t_df),
                      '2-sided independent t-test.')
            else:
                res_df.loc[n_rows+_i, "significant"] = 0
        else:
            print("n_levels are wrong for "+str(cols_[_i])) 
    if len(session_dict)!=0:
        res_df.loc[n_rows:, [list(session_dict.keys())[0]]] = list(session_dict.values())[0]
    else:
        res_df['session'] = 0
    return res_df

def categorize_vals(_x, norm_list, mode_):
    if mode_ == 3:
        if _x < norm_list[0]:
            return -1
        elif _x >= norm_list[0] and _x <= norm_list[1]:
            return 0
        elif _x > norm_list[1]:
            return 1 
        else:
            return np.nan
    else:
        if _x >= norm_list[0] and _x <= norm_list[1]:
            return 0
        elif _x < norm_list[0] or _x > norm_list[1]:
            return 1
        else:
            return np.nan

def report_data(df_, cols_, ref_col_, bins_, labels_):
    from scipy.stats import chisquare, chi2_contingency, fisher_exact
    from scipy.stats.contingency import association
    new_cols_=[]
    dims = []
    df_ = df_.loc[:, cols_].dropna(axis=0, how='any')
    for i_ in range(len(cols_)):
        col_ = cols_[i_]
        if len(bins_[i_])>1:
            new_col = col_+'_bin'
            df_.loc[:, new_col]=pd.cut(df_.loc[:, col_], bins=bins_[i_], labels=labels_[i_])
            new_cols_.append(new_col)
            dims.append(len(df_.loc[:, col_].unique()))
        else:
            new_cols_.append(col_)
            _tmp_list = df_.loc[:, col_].unique()
            if np.NaN in _tmp_list:
                dims.append(len(_tmp_list)-1)
            else:
                dims.append(len(_tmp_list))
    res_dict = {}
    if len(new_cols_) > 1:
        df_ = df_.loc[:,new_cols_].dropna(axis=0, how="any")
        tab_=pd.crosstab(df_.loc[:, new_cols_[0]], df_.loc[:, new_cols_[1]], margins=False)
        
        total = tab_.sum().sum()
    else:
        tab_=df_.groupby(new_cols_, observed=False).count()[ref_col_]
        total = tab_.sum()
    print(cols_,' in total:', total)
    if dims == [2,2]:
        stat, p, _no_, no_ = chi2_contingency(tab_)
    elif len(dims) > 1:
        stat = association(tab_, method="cramer")
        p=''
    else:
        stat = ''
        p=''

    tab_per = tab_.apply(lambda x: x/total)
    print('dims', dims, 'assosiation:',stat, 'p', p)
    print(tab_)
    print(tab_per)
    res_dict['cnt']=tab_
    res_dict['per']=tab_per
    return res_dict

# check normality
def check_res_norm(smf_res, title_, plot_=True):
    import statsmodels.api as sm
    import statsmodels.stats.diagnostic as diag
    import matplotlib.pyplot as plt
    if plot_:
        fig1 = sm.qqplot(smf_res.resid)
        plt.title(title_)
        plt.show()
    return diag.kstest_normal(smf_res.resid, dist='norm', pvalmethod='table')

## check data
def report_NAs(df_, tar_list=[], by_="col", plot_=False):
    res_dict={}
    if len(tar_list)==0:
        tar_list = list(df_.columns)
    if by_ =="col":
        for x_ in tar_list:
            na_list = df_[x_].isnull()
            na_num = na_list.sum()
            if na_num != 0:
                res_dict[x_]=na_num
                print(x_,":", na_num)
                for i_ in range(len(na_list)):
                    if na_list[i_]:
                        print(df_.index[i_])
    elif by_=="row":
        for x_ in list(df_.index):
            na_list = df_.loc[x_,:].isnull()
            na_num = na_list.sum()
            if na_num != 0:
                res_dict[x_]=na_num
                print(x_,":", na_num)
                for i_ in range(len(na_list)):
                    if na_list.iloc[i_]:
                        print(df_.columns[i_])
    else:
        print("report mode err...")
    if plot_:
        import matplotlib.pyplot as plt
        res_dict_sorted = dict(sorted(res_dict.items(), key=lambda item: item[1], reverse=True))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(list(res_dict_sorted.keys()), list(res_dict_sorted.values()), label=list(res_dict_sorted.keys()))
        plt.xticks(rotation = 90)
        plt.xticks(fontsize = 8)
    return res_dict

def ols_fit(data_, model_str_list, group_dict, report_model=True): 
    import statsmodels.formula.api as smf
    #from utils_.utils_private import check_res_norm
    col_list = ["test_var", "coef", "CI_l", "CI_h", "p_val", "std_err", "R2", "AIC", "BIC", "ks_val", "ks_pval","full_model","res",'KS_res']
    res_df = pd.DataFrame(columns=col_list)

    treatment_label = group_dict['treatment']
    control_label   = group_dict['control']
    res_col_name = 'C(group, Treatment(reference=\"'+control_label+'\"))[T.'+treatment_label+']'

    if len(model_str_list) <1:
        print("Nothing to test!")
    elif len(model_str_list) == 1:
        i_ = 0
        model_name = model_str_list[0].split("~")[0].strip(" ")
        mod = smf.ols(formula=model_str_list[0], data=data_)
        res = mod.fit()
        #
        res_df.loc[i_, "test_var"] = model_name
        res_df.loc[i_, "coef"] = res.params.loc[res_col_name]
        res_df.loc[i_, "CI_l"] = res.conf_int().loc[res_col_name, 0]
        res_df.loc[i_, "CI_h"] = res.conf_int().loc[res_col_name, 1]
        res_df.loc[i_, "p_val"] = res.pvalues.loc[res_col_name]
        res_df.loc[i_, "std_err"] = res.bse.loc[res_col_name]
        res_df.loc[i_, "R2"] = res.rsquared
        res_df.loc[i_, "AIC"] = res.aic
        res_df.loc[i_, "BIC"] = res.bic
        res_df.loc[i_, "full_model"] = model_str_list[0].replace(' ', '').replace('\n','')
        res_df.loc[i_, "res"] = res
        ks_res = check_res_norm(res, model_name + " residuals", plot_=False)
        res_df.loc[i_, "KS_res"] = ks_res
        res_df.loc[i_, "ks_val"] = ks_res[0]
        res_df.loc[i_, "ks_pval"] = ks_res[1]
    else:
        for i_ in range(len(model_str_list)):
            model_name = model_str_list[i_].split("~")[0].strip(" ")
            mod = smf.ols(formula=model_str_list[i_], data=data_)
            res = mod.fit()
            res_df.loc[i_, "test_var"] = model_name
            res_df.loc[i_, "coef"] = res.params.loc[res_col_name]
            res_df.loc[i_, "CI_l"] = res.conf_int().loc[res_col_name, 0]
            res_df.loc[i_, "CI_h"] = res.conf_int().loc[res_col_name, 1]
            res_df.loc[i_, "p_val"] = res.pvalues.loc[res_col_name]
            res_df.loc[i_, "std_err"] = res.bse.loc[res_col_name]
            res_df.loc[i_, "R2"] = res.rsquared
            res_df.loc[i_, "AIC"] = res.aic
            res_df.loc[i_, "BIC"] = res.bic
            res_df.loc[i_, "full_model"] = model_str_list[i_].replace(' ', '').replace('\n','')
            res_df.loc[i_, "res"] = res
            ks_res = check_res_norm(res, model_name + " residuals", plot_=False)
            res_df.loc[i_, "KS_res"] = ks_res
            res_df.loc[i_, "ks_val"] = ks_res[0]
            res_df.loc[i_, "ks_pval"] = ks_res[1]
    return res_df

def plot_es_list(res_df_, y_lim_, title_str, fig_size_=(8,18), ROTATION=15):
    import matplotlib.pyplot as plt
    # Preparing data
    x = []; y = []; ci_l = []; ci_h = []
    p_vals = []; x_labels=[]; std_err = []; R2_list= []; KS_plist = []
    for i_ in range(len(res_df_)):
        x_labels.append(res_df_.iloc[i_,0])
        x.append(i_*1+0.5)
        y.append(res_df_.iloc[i_,1])
        ci_l.append(res_df_.iloc[i_,1]-res_df_.iloc[i_,2])
        ci_h.append(res_df_.iloc[i_,3]-res_df_.iloc[i_,1])
        p_vals.append(res_df_.iloc[i_,4])
        std_err.append(res_df_.iloc[i_,5])
        R2_list.append(res_df_.iloc[i_,6]) 
        KS_plist.append(res_df_.iloc[i_,10])
    # plotting
    if len(y_lim_) != 2:
        y_adj = (max(res_df_.iloc[:,3])-min(res_df_.iloc[:,2]))*0.2 ## need some tuning, not working well
        y_lim_ = [min(ci_l)-y_adj, max(ci_h)+y_adj]
    colors = ["#A3E4D7", "#5499C7"]*int((len(x)/2))
    #ROTATION = 15
    ALPHA = 0.9
    #creating figure
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(4,1, figsize=fig_size_)
    ax1.scatter(x, y, s=30, color = colors )
    ax1.errorbar(x, y, yerr=[ci_h, ci_l], alpha=ALPHA, ecolor='black', capsize=4, fmt='o') # std_err
    ax1.hlines(xmin=0, xmax=max(x)+1, y=0, color='r', linestyles='--')
    ax1.set_xticks(x)
    ax1.set_ylim(y_lim_)
    ax1.set_xticklabels(x_labels, rotation = ROTATION)
    ax1.set_title(title_str+'\n'+'Coefficient of meditation effect (OLS) (redline: 0)')
    ax2.scatter(x, p_vals, alpha=ALPHA, color = colors)
    ax2.hlines(xmin=0, xmax=max(x)+1, y=0.05, color='r', linestyles='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation = ROTATION)
    ax2.set_ylabel('Treatment p-val(red line: 0.05)')
    ax3.scatter(x, R2_list, alpha=ALPHA, color = colors)
    ax3.hlines(xmin=0, xmax=max(x)+1, y=0.9, color='r', linestyles='--')
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, rotation = ROTATION)
    ax3.set_ylabel('Model R2 (red line: 0.9)')
    ax4.scatter(x, KS_plist, alpha=ALPHA, color = colors)
    ax4.hlines(xmin=0, xmax=max(x)+1, y=0.05, color='r', linestyles='--')
    ax4.set_xticks(x)
    ax4.set_xticklabels(x_labels, rotation = ROTATION)
    ax4.set_ylabel('Residual KS test p-val (red line: 0.05)')
    return 0

def demean_cols(df_, dmean_list):
    # adding new cols which removes the means of the original columns
    for x in dmean_list:
        df_.loc[:, x+"_rm"] = df_.loc[:, x] - df_.loc[:, x].mean()
    return df_
