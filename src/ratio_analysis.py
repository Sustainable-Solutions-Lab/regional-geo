"""
Gridded ratio analysis with error propagation.

Calculates ensemble-mean differences from control for T2 and loading variables,
normalizes by area-weighted sums to produce ratio fields, then creates
sorted/accumulated tables showing spatial contribution distribution.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Allow running as script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_variable, load_cell_area, EPISODES, ENSEMBLES, EMISSION_RATES

DOMAIN = 'd01'
TEMPORAL = 'allhr'
VARIABLES = ['T2', 'loading']
NON_CONTROL_RATES = ['1000', '10000', '100000']


def compute_ensemble_stats(variable, episode):
    """
    Load allhr data, time-average, and compute ensemble statistics.

    Parameters
    ----------
    variable : str
        Variable name ('T2' or 'loading')
    episode : str
        Episode ID ('240527' or '240727')

    Returns
    -------
    tuple of (dict, dict)
        means[rate] and ses[rate] as xarray.DataArrays with dims (south_north, west_east)
    """
    means = {}
    ses = {}

    for rate in EMISSION_RATES:
        # Load all ensembles for this episode and rate
        # Shape: (ensemble=3, Time=121, south_north=142, west_east=132)
        data = load_variable(variable, DOMAIN, TEMPORAL, episode=episode, emission_rate=rate)

        # Time-average: (ensemble, south_north, west_east)
        time_avg = data.mean(dim='Time')

        # Ensemble mean and SE
        ensemble_mean = time_avg.mean(dim='ensemble')
        ensemble_se = time_avg.std(dim='ensemble') / np.sqrt(len(ENSEMBLES))

        means[rate] = ensemble_mean
        ses[rate] = ensemble_se

    return means, ses


def compute_differences(means, ses):
    """
    Compute differences from control with propagated standard errors.

    Parameters
    ----------
    means : dict
        Dictionary mapping emission rate to mean field
    ses : dict
        Dictionary mapping emission rate to SE field

    Returns
    -------
    tuple of (dict, dict)
        diffs[rate] and diff_ses[rate] for non-control rates
    """
    ctl_mean = means['ctl']
    ctl_se = ses['ctl']

    diffs = {}
    diff_ses = {}

    for rate in NON_CONTROL_RATES:
        diffs[rate] = means[rate] - ctl_mean
        diff_ses[rate] = np.sqrt(ses[rate]**2 + ctl_se**2)

    return diffs, diff_ses


def compute_ratios(diffs, diff_ses, area):
    """
    Compute area-weighted sums and ratio fields.

    Parameters
    ----------
    diffs : dict
        Dictionary mapping emission rate to difference field
    diff_ses : dict
        Dictionary mapping emission rate to difference SE field
    area : xarray.DataArray
        Cell areas in km²

    Returns
    -------
    tuple of (dict, dict)
        ratios[rate] and ratio_ses[rate] with units 1/km²
    """
    ratios = {}
    ratio_ses = {}

    for rate in NON_CONTROL_RATES:
        d = diffs[rate]
        se_d = diff_ses[rate]

        # Area-weighted sum
        S = (d * area).sum()
        # SE of sum: sqrt(sum(SE_d^2 * area^2))
        se_S = np.sqrt((se_d**2 * area**2).sum())

        # Ratio field: d / S (units: 1/km²)
        ratios[rate] = d / S
        # Simplified SE: SE_d / |S|
        ratio_ses[rate] = se_d / np.abs(S)

    return ratios, ratio_ses


def average_ratios(ratios, ratio_ses):
    """
    Average ratio fields across emission rates.

    Parameters
    ----------
    ratios : dict
        Dictionary mapping emission rate to ratio field
    ratio_ses : dict
        Dictionary mapping emission rate to ratio SE field

    Returns
    -------
    tuple of (xarray.DataArray, xarray.DataArray)
        Mean ratio field and its SE
    """
    # Stack ratios: shape (3, south_north, west_east)
    ratio_stack = np.stack([ratios[rate].values for rate in NON_CONTROL_RATES], axis=0)
    se_stack = np.stack([ratio_ses[rate].values for rate in NON_CONTROL_RATES], axis=0)

    # Mean ratio
    r_mean = ratio_stack.mean(axis=0)

    # SE of mean: sqrt(sum(SE_i^2)) / n
    se_r_mean = np.sqrt((se_stack**2).sum(axis=0)) / len(NON_CONTROL_RATES)

    # Convert back to DataArray with coordinates
    template = ratios[NON_CONTROL_RATES[0]]
    r_mean_da = template.copy(data=r_mean)
    se_r_mean_da = template.copy(data=se_r_mean)

    return r_mean_da, se_r_mean_da


def build_sorted_table(r_mean, area):
    """
    Build sorted/accumulated table for ratio analysis.

    Parameters
    ----------
    r_mean : xarray.DataArray
        Mean ratio field (1/km²)
    area : xarray.DataArray
        Cell areas (km²)

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 7) with columns:
        [ratio, cell_area_km2, cumulative_contribution, cumulative_area_km2,
         cumulative_area_length_scale_km, inverse_ratio_area_km2, length_scale_km]
    """
    N = r_mean.size
    A = np.zeros((N, 7))

    A[:, 0] = r_mean.values.flatten()    # ratio (1/km²)
    A[:, 1] = area.values.flatten()       # cell area (km²)
    A[:, 2] = A[:, 0] * A[:, 1]           # contribution (dimensionless)
    A[:, 3] = A[:, 1].copy()              # area for cumulation
    A[:, 5] = 1.0 / A[:, 0]               # inverse of ratio (km²)
    A[:, 6] = np.sqrt(np.abs(A[:, 5]))    # length scale (km)

    # Sort descending by ratio (most positive first)
    idx = np.argsort(A[:, 0])[::-1]
    A = A[idx]

    # Cumulate columns 2 and 3
    A[:, 2] = np.cumsum(A[:, 2])
    A[:, 3] = np.cumsum(A[:, 3])
    A[:, 4] = np.sqrt(A[:, 3])            # cumulative area length scale (km)

    return A


def export_to_excel(table, variable, episode):
    """
    Export sorted table to Excel file.

    Parameters
    ----------
    table : numpy.ndarray
        Sorted/accumulated table from build_sorted_table
    variable : str
        Variable name ('T2' or 'loading')
    episode : str
        Episode ID ('240527' or '240727')
    """
    df = pd.DataFrame(
        table,
        columns=['ratio', 'cell_area_km2', 'cumulative_contribution', 'cumulative_area_km2',
                 'cumulative_area_length_scale_km', 'inverse_ratio_area_km2', 'length_scale_km']
    )

    output_path = f'data/output/{variable}_ratio_analysis_{episode}.xlsx'
    df.to_excel(output_path, index=False)
    print(f'Exported: {output_path}')


def create_combined_plots(tables):
    """
    Create a single PDF with all ratio analysis plots.

    Parameters
    ----------
    tables : dict
        Dictionary mapping (variable, episode) to sorted/accumulated table arrays.
        Columns: [ratio, cell_area_km2, cumulative_contribution, cumulative_area_km2,
                  cumulative_area_length_scale_km, inverse_ratio_area_km2, length_scale_km]
    """
    output_path = 'data/output/ratio_analysis.pdf'

    # Column indices for clarity
    COL_CUMULATIVE_CONTRIBUTION = 2
    COL_CUMULATIVE_AREA = 3
    COL_CUMULATIVE_LENGTH_SCALE = 4
    COL_INVERSE_RATIO_AREA = 5
    COL_LENGTH_SCALE = 6

    # Panel layout: rows = variables, cols = episodes
    panel_positions = [
        ('T2', '240527', 0, 0),
        ('T2', '240727', 0, 1),
        ('loading', '240527', 1, 0),
        ('loading', '240727', 1, 1),
    ]

    with PdfPages(output_path) as pdf:
        # Page 1: Cumulative Area vs Cumulative Contribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for variable, episode, row, col in panel_positions:
            table = tables[(variable, episode)]
            ax = axes[row, col]
            ax.plot(table[:, COL_CUMULATIVE_CONTRIBUTION], table[:, COL_CUMULATIVE_AREA],
                    'b-', linewidth=0.5)
            ax.set_xlabel('Cumulative Contribution')
            ax.set_ylabel('Cumulative Area (km²)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1e6)
        fig.suptitle('Cumulative Area vs Cumulative Contribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Inverse Ratio Area vs Cumulative Contribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for variable, episode, row, col in panel_positions:
            table = tables[(variable, episode)]
            ax = axes[row, col]
            ax.plot(table[:, COL_CUMULATIVE_CONTRIBUTION], table[:, COL_INVERSE_RATIO_AREA],
                    'g-', linewidth=0.5)
            ax.set_xlabel('Cumulative Contribution')
            ax.set_ylabel('Inverse Ratio Area (km²)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1e6)
        fig.suptitle('Inverse Ratio Area vs Cumulative Contribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Cumulative Length Scale vs Cumulative Contribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for variable, episode, row, col in panel_positions:
            table = tables[(variable, episode)]
            ax = axes[row, col]
            ax.plot(table[:, COL_CUMULATIVE_CONTRIBUTION], table[:, COL_CUMULATIVE_LENGTH_SCALE],
                    'r-', linewidth=0.5)
            ax.set_xlabel('Cumulative Contribution')
            ax.set_ylabel('Cumulative Area Length Scale (km)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1e3)
        fig.suptitle('Cumulative Length Scale vs Cumulative Contribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Cumulative Length Scale vs Length Scale
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for variable, episode, row, col in panel_positions:
            table = tables[(variable, episode)]
            ax = axes[row, col]
            ax.plot(table[:, COL_LENGTH_SCALE], table[:, COL_CUMULATIVE_LENGTH_SCALE],
                    'm-', linewidth=0.5)
            ax.set_xlabel('Length Scale (km)')
            ax.set_ylabel('Cumulative Area Length Scale (km)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1e3)
            ax.set_ylim(0, 1e3)
        fig.suptitle('Cumulative Length Scale vs Length Scale', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f'Created: {output_path}')


def analyze_variable_episode(variable, episode, area):
    """
    Run full analysis pipeline for one variable and episode.

    Parameters
    ----------
    variable : str
        Variable name ('T2' or 'loading')
    episode : str
        Episode ID ('240527' or '240727')
    area : xarray.DataArray
        Cell areas in km²

    Returns
    -------
    numpy.ndarray
        Sorted/accumulated table
    """
    print(f'\nAnalyzing {variable} for episode {episode}...')

    # Step 1-2: Load, time-average, compute ensemble stats
    print('  Computing ensemble statistics...')
    means, ses = compute_ensemble_stats(variable, episode)

    # Step 3: Compute differences from control
    print('  Computing differences from control...')
    diffs, diff_ses = compute_differences(means, ses)

    # Step 4-5: Compute ratios
    print('  Computing ratio fields...')
    ratios, ratio_ses = compute_ratios(diffs, diff_ses, area)

    # Step 6: Average ratios across emission rates
    print('  Averaging across emission rates...')
    r_mean, se_r_mean = average_ratios(ratios, ratio_ses)

    # Step 7: Build sorted table
    print('  Building sorted table...')
    table = build_sorted_table(r_mean, area)

    # Step 8: Export to Excel
    export_to_excel(table, variable, episode)

    # Print verification info
    print(f'  Final cumulative contribution: {table[-1, 2]:.6f}')
    print(f'  Final cumulative area: {table[-1, 3]:.2f} km²')

    return table


def main():
    """Run full ratio analysis pipeline for all variables and episodes."""
    print('=' * 60)
    print('Gridded Ratio Analysis with Error Propagation')
    print('=' * 60)

    # Load cell areas once
    print(f'\nLoading cell areas for domain {DOMAIN}...')
    area = load_cell_area(DOMAIN)
    total_area = area.sum().values
    print(f'  Grid size: {area.shape[0]} x {area.shape[1]} = {area.size} cells')
    print(f'  Total area: {total_area:.2f} km²')

    # Analyze all combinations
    results = {}
    for variable in VARIABLES:
        for episode in EPISODES:
            key = (variable, episode)
            results[key] = analyze_variable_episode(variable, episode, area)

    # Create combined PDF with all plots
    print('\nCreating combined plots...')
    create_combined_plots(results)

    print('\n' + '=' * 60)
    print('Analysis complete!')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
