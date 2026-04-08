# -*- coding: utf-8 -*-
"""
This is the utils lib for imaging preprocessing and processing created by Qing Wang (Vincent)."
2026.3.17
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
# vis
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

def load_subject_timeseries(sub_id, master_file_df, drop_vols=10):
    """Fetches the raw TSV for a specific subject"""
    # Find the file path from your original file_df
    confound_path = master_file_df[master_file_df['participant_id'] == sub_id]['confound_file'].values[0]
    df = pd.read_csv(confound_path, sep='\t').iloc[drop_vols:]
    
    # Drop initial NaNs
    valid_idx = ~df['framewise_displacement'].isna() & ~df['std_dvars'].isna()
    return df[valid_idx]

# Cohort-Level QC Dashboard
def plot_cohort_qc_dashboard(qc_df):
    """
    Generates an interactive Macro QA dashboard (Mean FD vs Mean DVARS) for an entire cohort.
    Expects a DataFrame with columns: ['sub', 'group', 'mean_fd', 'mean_dvars', 'vols_rem_DUAL']
    """
    # Clean missing data
    df_macro = qc_df.dropna(subset=['mean_fd', 'mean_dvars']).copy()

    # Initialize Plotly FigureWidget
    fig_cohort = go.FigureWidget()

    # Dynamically assign colors to however many groups exist in this cohort
    groups = df_macro['group'].unique()
    colors = px.colors.qualitative.Plotly
    
    for i, group_name in enumerate(groups):
        group_data = df_macro[df_macro['group'] == group_name]
        color = colors[i % len(colors)]
        
        fig_cohort.add_scatter(
            x=group_data['mean_fd'], 
            y=group_data['mean_dvars'],
            mode='markers',
            name=str(group_name),
            marker=dict(size=8, color=color, opacity=0.8, line=dict(width=1, color='white')),
            text=[f"Sub: {row['sub']}<br>Group: {row['group']}<br>Mean FD: {row['mean_fd']:.3f}<br>Mean DVARS: {row['mean_dvars']:.3f}<br>Vols Retained (Dual): {row['vols_rem_DUAL']}" 
                  for _, row in group_data.iterrows()],
            hoverinfo='text'
        )

    fig_cohort.add_vline(x=0.25, line_width=2, line_dash="dash", line_color="black")
    fig_cohort.add_hline(y=1.3, line_width=2, line_dash="dash", line_color="black")

    fig_cohort.update_layout(
        title='Cohort Quality Control: Mean FD vs. Mean DVARS',
        xaxis_title='Mean Framewise Displacement (FD)',
        yaxis_title='Mean Standardized DVARS',
        template='plotly_white',
        width=900, height=550,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # UI Widgets
    fd_slider = widgets.FloatSlider(value=0.5, min=0.05, max=2, step=0.01, description='FD:')
    dvars_slider = widgets.FloatSlider(value=1.5, min=0.8, max=3, step=0.05, description='DVARS:')
    output_text = widgets.HTML()

    # Update Logic
    def update_plot(*args):
        fd_limit = fd_slider.value
        dvars_limit = dvars_slider.value
        
        fig_cohort.layout.shapes[0].x0 = fig_cohort.layout.shapes[0].x1 = fd_limit
        fig_cohort.layout.shapes[1].y0 = fig_cohort.layout.shapes[1].y1 = dvars_limit
        
        survivors = df_macro[(df_macro['mean_fd'] <= fd_limit) & (df_macro['mean_dvars'] <= dvars_limit)]
        output_text.value = f"<h3 style='color: #1f77b4'>Subjects in Safe Zone: {len(survivors)} / {len(df_macro)}</h3>"
        
        with fig_cohort.batch_update():
            for j, group_name in enumerate(groups):
                group_data = df_macro[df_macro['group'] == group_name]
                opacities = [0.8 if (row['mean_fd'] <= fd_limit and row['mean_dvars'] <= dvars_limit) else 0.1 
                             for _, row in group_data.iterrows()]
                fig_cohort.data[j].marker.opacity = opacities

    fd_slider.observe(update_plot, 'value')
    dvars_slider.observe(update_plot, 'value')
    update_plot()

    # Display the dashboard
    display(widgets.VBox([widgets.HBox([fd_slider, dvars_slider]), output_text, fig_cohort]))

# Subject-Level Scrubbing Simulator
def plot_subject_scrubbing_simulator(sub_id, master_file_df, drop_vols=10):
    """
    Generates a frame-by-frame scrubbing simulator for a specific subject.
    Expects the subject ID and the original DataFrame containing the 'confound_file' paths.
    """
    # Fetch data
    try:
        ts_df = load_subject_timeseries(sub_id, master_file_df)
    except Exception as e:
        print(f"Error loading data for {sub_id}: {e}")
        return
    
    total_frames = len(ts_df)
    fig_micro = go.FigureWidget()

    fig_micro.add_scatter(
        x=ts_df['framewise_displacement'], y=ts_df['std_dvars'], mode='markers',
        marker=dict(size=8, color='#1f77b4', opacity=0.7, line=dict(width=1, color='white')),
        text=[f"TR Index: {i}<br>FD: {f:.3f}<br>DVARS: {d:.3f}" 
              for i, (f, d) in zip(ts_df.index, zip(ts_df['framewise_displacement'], ts_df['std_dvars']))],
        hoverinfo='text'
    )

    fig_micro.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="gray")
    fig_micro.add_hline(y=1.5, line_width=2, line_dash="dash", line_color="gray")

    fig_micro.update_layout(
        title=f'Frame-by-Frame Simulator: Subject {sub_id}',
        xaxis_title='Framewise Displacement (FD)', yaxis_title='Standardized DVARS',
        template='plotly_white', width=800, height=500
    )

    # UI Elements
    strat_dropdown = widgets.Dropdown(options=['FD-Only', 'Dual-Threshold'], value='Dual-Threshold', description='Strategy:')
    fd_slider = widgets.FloatSlider(value=0.5, min=0.1, max=2, step=0.05, description='FD Cutoff:')
    dvars_slider = widgets.FloatSlider(value=1.5, min=1.0, max=3.0, step=0.1, description='DVARS Cutoff:')
    output_text = widgets.HTML()

    def update_micro(*args):
        fd_t, dvars_t, strat = fd_slider.value, dvars_slider.value, strat_dropdown.value
        
        fig_micro.layout.shapes[0].x0 = fig_micro.layout.shapes[0].x1 = fd_t
        fig_micro.layout.shapes[1].y0 = fig_micro.layout.shapes[1].y1 = dvars_t
        fig_micro.layout.shapes[1].opacity = 0 if strat == 'FD-Only' else 1
        dvars_slider.disabled = (strat == 'FD-Only')
        
        if strat == 'FD-Only':
            scrubbed = ts_df['framewise_displacement'] > fd_t
        else:
            scrubbed = (ts_df['framewise_displacement'] > fd_t) & (ts_df['std_dvars'] > dvars_t)
            
        retained = (~scrubbed).sum()
        colors = ['#d62728' if s else '#1f77b4' for s in scrubbed]
        
        with fig_micro.batch_update():
            fig_micro.data[0].marker.color = colors
            output_text.value = f"<h3 style='color: {'#1f77b4' if retained >= 200 else '#d62728'}'>Frames Retained: {retained} / {total_frames}</h3>"

    strat_dropdown.observe(update_micro, 'value')
    fd_slider.observe(update_micro, 'value')
    dvars_slider.observe(update_micro, 'value')
    update_micro()

    display(widgets.VBox([widgets.HBox([strat_dropdown, output_text]), widgets.HBox([fd_slider, dvars_slider]), fig_micro]))

def plot_attrition_curve(qc_df, min_vols=100, max_vols=250, step=5, tr_seconds=2.0, save_html_path=None):
    """
    Loops through volume thresholds and plots the survival curve for the cohort.
    Can export a standalone interactive HTML file for easy sharing.
    """
    print(f"Calculating attrition from {min_vols} to {max_vols} volumes...")
    
    # 1. Loop through thresholds and calculate survival
    results = []
    for target in range(min_vols, max_vols + step, step):
        pass_fd = (qc_df['vols_rem_FD05'] >= target).sum()
        pass_dual = (qc_df['vols_rem_DUAL'] >= target).sum()
        salvaged = pass_dual - pass_fd
        
        results.append({
            'Target_Volumes': target,
            'Minutes_of_Data': (target * tr_seconds) / 60.0,
            'FD_Only_Survival': pass_fd,
            'Dual_Survival': pass_dual,
            'Subjects_Salvaged': salvaged
        })
        
    curve_df = pd.DataFrame(results)
    
    # 2. Initialize the Plotly Figure
    fig = go.Figure()

    # Add FD-Only Line
    fig.add_trace(go.Scatter(
        x=curve_df['Target_Volumes'], y=curve_df['FD_Only_Survival'],
        mode='lines+markers',
        name='Standard (FD > 0.5mm Only)',
        line=dict(color='#d62728', width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='<b>Target TRs:</b> %{x}<br><b>Retained:</b> %{y}<extra></extra>'
    ))

    # Add Dual-Threshold Line
    fig.add_trace(go.Scatter(
        x=curve_df['Target_Volumes'], y=curve_df['Dual_Survival'],
        mode='lines+markers',
        name='Advanced (FD > 0.5 AND DVARS > 1.5)',
        line=dict(color='#1f77b4', width=4),
        marker=dict(size=8),
        hovertemplate='<b>Target TRs:</b> %{x}<br><b>Retained:</b> %{y}<extra></extra>'
    ))

    # 3. Add Clinical Gold Standard Highlights
    fig.add_vline(x=150, line_width=1, line_dash="dot", line_color="green", annotation_text=" 5 Mins", annotation_position="top right")
    fig.add_vline(x=200, line_width=1, line_dash="dot", line_color="green", annotation_text=" 6.67 Mins", annotation_position="top right")

    # 4. Format the Layout
    fig.update_layout(
        title='Cohort Attrition Curve: Volume Threshold vs. Subject Retention',
        xaxis_title='Minimum Usable Volumes Required',
        yaxis_title='Total Subjects Retained',
        hovermode='x unified', 
        template='plotly_white',
        legend=dict(x=0.02, y=0.05, bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1),
        width=900, height=550
    )

    # 5. Display and Export
    fig.show()
    
    if save_html_path:
        fig.write_html(save_html_path)
        print(f"\n✅ Interactive snapshot successfully saved to: {save_html_path}")
    
    return curve_df