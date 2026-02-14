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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

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
        Array of shape (N, 9) with columns:
        [ratio, cell_area_km2, cumulative_contribution, cumulative_area_km2,
         cumulative_area_length_scale_km, inverse_ratio_area_km2, length_scale_km,
         south_north_idx, west_east_idx]
    """
    n_south_north, n_west_east = r_mean.shape
    N = r_mean.size
    A = np.zeros((N, 9))

    A[:, 0] = r_mean.values.flatten()    # ratio (1/km²)
    A[:, 1] = area.values.flatten()       # cell area (km²)
    A[:, 2] = A[:, 0] * A[:, 1]           # contribution (dimensionless)
    A[:, 3] = A[:, 1].copy()              # area for cumulation
    A[:, 5] = 1.0 / A[:, 0]               # inverse of ratio (km²)
    A[:, 6] = np.sqrt(np.abs(A[:, 5]))    # length scale (km)

    # Create grid indices for each flattened cell
    south_north_idx, west_east_idx = np.meshgrid(
        np.arange(n_south_north), np.arange(n_west_east), indexing='ij'
    )
    A[:, 7] = south_north_idx.flatten()
    A[:, 8] = west_east_idx.flatten()

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
                 'cumulative_area_length_scale_km', 'inverse_ratio_area_km2', 'length_scale_km',
                 'south_north_idx', 'west_east_idx']
    )

    output_path = f'data/output/{variable}_ratio_analysis_{episode}.xlsx'
    df.to_excel(output_path, index=False)
    print(f'Exported: {output_path}')


def build_cumulative_contribution_map(table, grid_shape):
    """
    Reconstruct 2D cumulative contribution map from sorted table.

    Parameters
    ----------
    table : numpy.ndarray
        Sorted table from build_sorted_table with columns including
        cumulative_contribution (col 2), south_north_idx (col 7), west_east_idx (col 8)
    grid_shape : tuple
        Shape of the output grid (n_south_north, n_west_east)

    Returns
    -------
    numpy.ndarray
        2D array of cumulative contribution values at each grid cell
    """
    cumulative_map = np.zeros(grid_shape)
    south_north_idx = table[:, 7].astype(int)
    west_east_idx = table[:, 8].astype(int)
    cumulative_map[south_north_idx, west_east_idx] = table[:, 2]
    return cumulative_map


def get_wrf_projection_and_coords(domain):
    """
    Load Lambert Conformal projection and coordinates from WRF geo_em file.

    Parameters
    ----------
    domain : str
        WRF domain identifier (e.g., 'd01')

    Returns
    -------
    tuple
        (projection, xlat, xlong) where projection is a cartopy CRS,
        and xlat/xlong are 2D coordinate arrays
    """
    geo_path = f'data/input/WRFPOST/geo_em.{domain}.nc'
    ds = xr.open_dataset(geo_path)

    projection = ccrs.LambertConformal(
        central_longitude=float(ds.attrs['CEN_LON']),
        central_latitude=float(ds.attrs['CEN_LAT']),
        standard_parallels=(float(ds.attrs['TRUELAT1']),)
    )
    xlat = ds['XLAT_M'].isel(Time=0).values
    xlong = ds['XLONG_M'].isel(Time=0).values
    return projection, xlat, xlong


def plot_cumulative_contribution_map(ax, cumulative_map, xlat, xlong, projection, title):
    """
    Plot filled contours of cumulative contribution with contour lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with cartopy projection to plot on
    cumulative_map : numpy.ndarray
        2D array of cumulative contribution values
    xlat : numpy.ndarray
        2D latitude coordinate array
    xlong : numpy.ndarray
        2D longitude coordinate array
    projection : cartopy.crs.CRS
        Map projection for the data
    title : str
        Title for the subplot

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The filled contour object (for colorbar)
    """
    data_crs = ccrs.PlateCarree()
    levels = np.arange(0, 1.1, 0.1)

    cf = ax.contourf(xlong, xlat, cumulative_map, levels=levels,
                     cmap='viridis', transform=data_crs)
    ax.contour(xlong, xlat, cumulative_map, levels=levels,
               colors='black', linewidths=0.5, transform=data_crs)

    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray')
    ax.set_title(title)
    return cf


def create_combined_plots(results, area):
    """
    Create a single PDF with all ratio analysis plots.

    Parameters
    ----------
    results : dict
        Dictionary mapping (variable, episode) to dict of tables.
        Each inner dict maps rate keys ('mean', '1000', '10000', '100000') to sorted tables.
        Table columns: [ratio, cell_area_km2, cumulative_contribution, cumulative_area_km2,
                        cumulative_area_length_scale_km, inverse_ratio_area_km2, length_scale_km,
                        south_north_idx, west_east_idx]
    area : xarray.DataArray
        Cell areas in km² (used to extract grid shape for maps)
    """
    output_path = 'data/output/ratio_analysis.pdf'

    # Column indices for clarity
    COL_CUMULATIVE_CONTRIBUTION = 2
    COL_CUMULATIVE_AREA = 3
    COL_CUMULATIVE_LENGTH_SCALE = 4
    COL_INVERSE_RATIO_AREA = 5
    COL_LENGTH_SCALE = 6

    # Line styles: individual rates first (thin), then mean (thick) so mean is on top
    LINE_STYLES = {
        '1000': {'color': 'blue', 'linewidth': 0.5, 'alpha': 0.7, 'label': 'r1000'},
        '10000': {'color': 'green', 'linewidth': 0.5, 'alpha': 0.7, 'label': 'r10000'},
        '100000': {'color': 'red', 'linewidth': 0.5, 'alpha': 0.7, 'label': 'r100000'},
        'mean': {'color': 'black', 'linewidth': 2, 'alpha': 1.0, 'label': 'Mean'},
    }
    RATE_ORDER = ['1000', '10000', '100000', 'mean']  # mean last so it's on top

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
            tables = results[(variable, episode)]
            ax = axes[row, col]
            for rate_key in RATE_ORDER:
                table = tables[rate_key]
                style = LINE_STYLES[rate_key]
                ax.plot(table[:, COL_CUMULATIVE_CONTRIBUTION], table[:, COL_CUMULATIVE_AREA],
                        **style)
            ax.set_xlabel('Cumulative Contribution')
            ax.set_ylabel('Cumulative Area (km²)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1e6)
            ax.legend(loc='lower right', fontsize=8)
        fig.suptitle('Cumulative Area vs Cumulative Contribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Inverse Ratio Area vs Cumulative Contribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for variable, episode, row, col in panel_positions:
            tables = results[(variable, episode)]
            ax = axes[row, col]
            for rate_key in RATE_ORDER:
                table = tables[rate_key]
                style = LINE_STYLES[rate_key]
                ax.plot(table[:, COL_CUMULATIVE_CONTRIBUTION], table[:, COL_INVERSE_RATIO_AREA],
                        **style)
            ax.set_xlabel('Cumulative Contribution')
            ax.set_ylabel('Inverse Ratio Area (km²)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1e6)
            ax.legend(loc='upper right', fontsize=8)
        fig.suptitle('Inverse Ratio Area vs Cumulative Contribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Cumulative Length Scale vs Cumulative Contribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for variable, episode, row, col in panel_positions:
            tables = results[(variable, episode)]
            ax = axes[row, col]
            for rate_key in RATE_ORDER:
                table = tables[rate_key]
                style = LINE_STYLES[rate_key]
                ax.plot(table[:, COL_CUMULATIVE_CONTRIBUTION], table[:, COL_CUMULATIVE_LENGTH_SCALE],
                        **style)
            ax.set_xlabel('Cumulative Contribution')
            ax.set_ylabel('Cumulative Area Length Scale (km)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1e3)
            ax.legend(loc='lower right', fontsize=8)
        fig.suptitle('Cumulative Length Scale vs Cumulative Contribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Cumulative Length Scale vs Length Scale
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for variable, episode, row, col in panel_positions:
            tables = results[(variable, episode)]
            ax = axes[row, col]
            for rate_key in RATE_ORDER:
                table = tables[rate_key]
                style = LINE_STYLES[rate_key]
                ax.plot(table[:, COL_LENGTH_SCALE], table[:, COL_CUMULATIVE_LENGTH_SCALE],
                        **style)
            ax.set_xlabel('Length Scale (km)')
            ax.set_ylabel('Cumulative Area Length Scale (km)')
            ax.set_title(f'{variable} — {episode}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1e3)
            ax.set_ylim(0, 1e3)
            ax.legend(loc='lower right', fontsize=8)
        fig.suptitle('Cumulative Length Scale vs Length Scale', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Load WRF projection and coordinates for maps
        grid_shape = area.shape
        projection, xlat, xlong = get_wrf_projection_and_coords(DOMAIN)

        # Map pages: one page per variable×episode, panels for each rate
        rate_panel_configs = [
            ('mean', 'Mean', 0, 0),
            ('1000', 'r1000', 0, 1),
            ('10000', 'r10000', 1, 0),
            ('100000', 'r100000', 1, 1),
        ]

        for variable in VARIABLES:
            for episode in EPISODES:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                         subplot_kw={'projection': projection})
                cf = None
                tables = results[(variable, episode)]
                for rate_key, rate_label, row, col in rate_panel_configs:
                    table = tables[rate_key]
                    cumulative_map = build_cumulative_contribution_map(table, grid_shape)
                    # For T2, use absolute value since effects differ in sign
                    if variable == 'T2':
                        cumulative_map = np.abs(cumulative_map)
                    ax = axes[row, col]
                    cf = plot_cumulative_contribution_map(
                        ax, cumulative_map, xlat, xlong, projection,
                        rate_label
                    )
                fig.suptitle(f'{variable} — {episode}: Cumulative Contribution Maps',
                             fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0.08, 1, 1])
                cbar = fig.colorbar(cf, ax=axes, orientation='horizontal', fraction=0.04,
                                    pad=0.02, shrink=0.5)
                cbar.set_label('Fraction of total loading within region', fontsize=11)
                pdf.savefig(fig)
                plt.close(fig)

        # Zoomed map pages for T2: one page per episode
        zoom_levels = np.arange(0, 0.22, 0.02)

        for episode in EPISODES:
            # Compute zoom extent from mean cumulative map
            # Find grid cells where cumulative contribution <= 0.4 in the mean
            mean_table = results[('T2', episode)]['mean']
            mean_map = build_cumulative_contribution_map(mean_table, grid_shape)
            mean_map = np.abs(mean_map)

            # Find cells contributing to first 40% of total
            mask = mean_map <= 0.4
            lat_in_region = xlat[mask]
            lon_in_region = xlong[mask]
            lat_min, lat_max = lat_in_region.min(), lat_in_region.max()
            lon_min, lon_max = lon_in_region.min(), lon_in_region.max()
            # Add small buffer
            lat_buffer = (lat_max - lat_min) * 0.1
            lon_buffer = (lon_max - lon_min) * 0.1
            extent = [lon_min - lon_buffer, lon_max + lon_buffer,
                      lat_min - lat_buffer, lat_max + lat_buffer]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                     subplot_kw={'projection': projection})
            cf = None
            tables = results[('T2', episode)]
            for rate_key, rate_label, row, col in rate_panel_configs:
                table = tables[rate_key]
                cumulative_map = build_cumulative_contribution_map(table, grid_shape)
                cumulative_map = np.abs(cumulative_map)
                ax = axes[row, col]
                data_crs = ccrs.PlateCarree()
                cf = ax.contourf(xlong, xlat, cumulative_map, levels=zoom_levels,
                                 cmap='viridis', transform=data_crs, extend='max')
                ax.contour(xlong, xlat, cumulative_map, levels=zoom_levels,
                           colors='black', linewidths=0.5, transform=data_crs)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray')
                ax.set_extent(extent, crs=data_crs)
                ax.set_title(rate_label)

            fig.suptitle(f'T2 — {episode}: Zoomed Cumulative Contribution (0–0.2)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            cbar = fig.colorbar(cf, ax=axes, orientation='horizontal', fraction=0.04,
                                pad=0.02, shrink=0.5)
            cbar.set_label('Fraction of total T2 change within region', fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)

    print(f'Created: {output_path}')


def export_peak_locations(results, domain):
    """
    Export latitude and longitude of peak ratio locations to CSV.

    Parameters
    ----------
    results : dict
        Dictionary mapping (variable, episode) to dict of tables
    domain : str
        WRF domain identifier for loading coordinates
    """
    _, xlat, xlong = get_wrf_projection_and_coords(domain)

    rows = []
    rate_keys = ['mean', '1000', '10000', '100000']

    for variable in VARIABLES:
        for episode in EPISODES:
            tables = results[(variable, episode)]
            for rate_key in rate_keys:
                table = tables[rate_key]
                # First row has highest ratio (sorted descending)
                south_north_idx = int(table[0, 7])
                west_east_idx = int(table[0, 8])
                lat = xlat[south_north_idx, west_east_idx]
                lon = xlong[south_north_idx, west_east_idx]
                ratio = table[0, 0]

                rows.append({
                    'variable': variable,
                    'episode': episode,
                    'rate': rate_key,
                    'latitude': lat,
                    'longitude': lon,
                    'ratio': ratio,
                    'south_north_idx': south_north_idx,
                    'west_east_idx': west_east_idx
                })

    df = pd.DataFrame(rows)
    output_path = 'data/output/peak_locations.csv'
    df.to_csv(output_path, index=False)
    print(f'Exported: {output_path}')


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
    dict
        Dictionary mapping rate keys ('mean', '1000', '10000', '100000') to sorted tables
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

    # Step 7: Build sorted tables for mean and each rate
    print('  Building sorted tables...')
    tables = {}
    tables['mean'] = build_sorted_table(r_mean, area)
    for rate in NON_CONTROL_RATES:
        tables[rate] = build_sorted_table(ratios[rate], area)

    # Step 8: Export mean table to Excel
    export_to_excel(tables['mean'], variable, episode)

    # Print verification info
    print(f'  Final cumulative contribution: {tables["mean"][-1, 2]:.6f}')
    print(f'  Final cumulative area: {tables["mean"][-1, 3]:.2f} km²')

    return tables


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
    create_combined_plots(results, area)

    # Export peak locations to CSV
    print('\nExporting peak locations...')
    export_peak_locations(results, DOMAIN)

    print('\n' + '=' * 60)
    print('Analysis complete!')
    print('=' * 60)

    return results


if __name__ == '__main__':
    main()
