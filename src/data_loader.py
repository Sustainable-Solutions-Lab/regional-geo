"""
Utility for loading WRF-Chem simulation data.

Loads NetCDF files and combines them across experimental dimensions
(Episode, Ensemble, Emission_Rate) as needed.
"""

import os
import xarray as xr

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', 'WRFPOST')
GEO_EM_PATH = os.path.join(DATA_DIR, 'geo_em.{domain}.nc')

EPISODES = ['240527', '240727']
ENSEMBLES = ['e1', 'e2', 'e3']
EMISSION_RATES = ['ctl', '1000', '10000', '100000']


def get_case_dir(episode, ensemble, emission_rate):
    """Get the directory path for a specific case."""
    if emission_rate == 'ctl':
        dirname = f'D9_{episode}_{ensemble}_ctl'
    else:
        dirname = f'D9_{episode}_{ensemble}_{emission_rate}_5x5'
    return os.path.join(DATA_DIR, dirname)


def load_variable(variable, domain, temporal, episode=None, ensemble=None, emission_rate=None):
    """
    Load WRF-Chem variable data.

    Parameters
    ----------
    variable : str
        Variable name (e.g., 'T2', 'SWDNT', 'so4_a01')
    domain : str
        Domain identifier ('d01' or 'd02')
    temporal : str
        Temporal resolution ('allhr' or 'tmean5d')
    episode : str, optional
        Episode ID ('240527' or '240727'). If None, loads all episodes.
    ensemble : str, optional
        Ensemble ID ('e1', 'e2', or 'e3'). If None, loads all ensembles.
    emission_rate : str, optional
        Emission rate ('ctl', '1000', '10000', or '100000'). If None, loads all rates.

    Returns
    -------
    xarray.DataArray
        Data array with dimensions for any unspecified experimental factors.
    """
    episodes = [episode] if episode else EPISODES
    ensembles = [ensemble] if ensemble else ENSEMBLES
    emission_rates = [emission_rate] if emission_rate else EMISSION_RATES

    filename = f'{temporal}_{domain}_{variable}.nc'

    data_by_episode = []
    for ep in episodes:
        data_by_ensemble = []
        for ens in ensembles:
            data_by_emission = []
            for em in emission_rates:
                case_dir = get_case_dir(ep, ens, em)
                filepath = os.path.join(case_dir, filename)
                ds = xr.open_dataset(filepath)
                da = ds[variable]
                data_by_emission.append(da)

            if len(emission_rates) > 1:
                combined = xr.concat(data_by_emission, dim='emission_rate')
                combined['emission_rate'] = emission_rates
            else:
                combined = data_by_emission[0]
            data_by_ensemble.append(combined)

        if len(ensembles) > 1:
            combined = xr.concat(data_by_ensemble, dim='ensemble')
            combined['ensemble'] = ensembles
        else:
            combined = data_by_ensemble[0]
        data_by_episode.append(combined)

    if len(episodes) > 1:
        result = xr.concat(data_by_episode, dim='episode')
        result['episode'] = episodes
    else:
        result = data_by_episode[0]

    return result


def load_cell_area(domain):
    """
    Return cell areas in km² for a WRF domain.

    Uses the WRF map factor method: area = (DX * DY) / (MAPFAC_M ** 2)

    Parameters
    ----------
    domain : str
        Domain identifier ('d01' or 'd02')

    Returns
    -------
    xarray.DataArray
        Cell areas in km² with dims (south_north, west_east) and
        coordinates (XLAT, XLONG).
    """
    geo_path = GEO_EM_PATH.format(domain=domain)
    ds = xr.open_dataset(geo_path)

    dx = ds.attrs['DX']
    dy = ds.attrs['DY']
    mapfac_m = ds['MAPFAC_M'].isel(Time=0)

    # Compute area in km² (DX, DY are in meters)
    area = (dx * dy) / (mapfac_m ** 2) / 1e6

    area = area.assign_coords({
        'XLAT': ds['XLAT_M'].isel(Time=0),
        'XLONG': ds['XLONG_M'].isel(Time=0),
    })
    area.name = 'cell_area'
    area.attrs['units'] = 'km²'
    area.attrs['long_name'] = 'Grid cell area'

    return area


def list_available_cases():
    """List all available case directories."""
    cases = []
    for entry in os.listdir(DATA_DIR):
        if entry.startswith('D9_') and os.path.isdir(os.path.join(DATA_DIR, entry)):
            cases.append(entry)
    return sorted(cases)


def list_variables(case_dir, domain, temporal):
    """List available variables for a given case, domain, and temporal resolution."""
    case_path = os.path.join(DATA_DIR, case_dir)
    prefix = f'{temporal}_{domain}_'
    variables = []
    for f in os.listdir(case_path):
        if f.startswith(prefix) and f.endswith('.nc'):
            var = f[len(prefix):-3]
            variables.append(var)
    return sorted(variables)
