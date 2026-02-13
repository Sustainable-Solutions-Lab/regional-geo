# regional-geo

Analysis of WRF regional climate model simulations, exploring the effects of sulfate aerosol injection on regional climate.

## Setup

```bash
pip install -r requirements.txt
```

## Experimental Design

The simulations explore stratospheric aerosol injection scenarios using the WRF-Chem regional climate model.

### Dimensions of the Experiment

| Dimension | Values | Description |
|-----------|--------|-------------|
| **Episode** | `240527`, `240727` | Two seasonal periods: May 2024 (dry season) and July 2024 (wet season) |
| **Ensemble** | `e1`, `e2`, `e3` | Three ensemble members per episode for uncertainty quantification |
| **Emission Rate** | `ctl`, `1000`, `10000`, `100000` | SO₂ injection rate in t/h (tons per hour): control (0), 1 kt/h, 10 kt/h, 100 kt/h |
| **Injection Region** | `5x5` | 5×5 grid cell injection region in domain d02 |

Total expected cases: 2 episodes × 3 ensembles × 4 emission rates = 24 cases

## Data

WRF post-processed output files (NetCDF format) are stored in `data/input/`.

### Directory Structure

```
data/input/
└── WRFPOST/                    # Main data directory
    ├── README.md               # Detailed documentation from data provider
    ├── geo_em.d01.nc           # Domain 1 geography (land use, terrain, coordinates)
    ├── geo_em.d02.nc           # Domain 2 geography
    └── D9_{episode}_{ensemble}_{emission}/   # Case directories
        ├── allhr_d0{1,2}_{VAR}.nc            # Hourly data (121 timesteps)
        └── tmean5d_d0{1,2}_{VAR}.nc          # 5-day mean data
```

### Directory Naming Convention

```
D9_{episodeID}_{ensembleID}_{emissionRate}[_5x5]
```

Examples:
- `D9_240527_e1_ctl` — May episode, ensemble 1, control run (no injection)
- `D9_240727_e2_10000_5x5` — July episode, ensemble 2, 10 kt/h injection, 5×5 grid

### File Naming Convention

```
{temporal}_{domain}_{VARIABLE}.nc
```

| Component | Values | Description |
|-----------|--------|-------------|
| `temporal` | `allhr`, `tmean5d` | Hourly (121 timesteps over 5 days) or 5-day temporal mean |
| `domain` | `d01`, `d02` | Outer domain (coarser) or inner domain (finer resolution) |
| `VARIABLE` | `T2`, `OLR`, etc. | Variable name in uppercase |

### File Dimension Patterns

Each NetCDF file contains a single variable with dimensions depending on the file type and variable:

| File Type | Dimensions | Variables |
|-----------|------------|-----------|
| `allhr_*` 2D | (Time, south_north, west_east) | T2, ALBEDO, OLR, PBLH, LH, HFX, QFX, SWDNT, SWUPT, SWDNB, SWUPB, LWDNT, LWUPT, LWDNB, LWUPB, SWCF, LWCF, and clear-sky variants (*C) |
| `allhr_*` 3D | (Time, bottom_top, south_north, west_east) | T, P, PB, U, V, W, QVAPOR, QCLOUD, QICE, QNDROP, CLDFRA, EXTCOF55, PM2_5_DRY, PM10, so4_a, so4_a01–so4_a04 |
| `tmean5d_*` 2D | (south_north, west_east) | Same 2D variables as allhr, but time-averaged |
| `tmean5d_*` 3D | (bottom_top, south_north, west_east) | Same 3D variables as allhr, but time-averaged |

Dimension sizes:
- `Time`: 121 hourly timesteps (5 days starting after 2-day spin-up)
- `bottom_top`: 49 vertical levels
- `south_north` / `west_east`: Grid dimensions (see below)

## Grid Dimensions

| Domain | Grid Size | Description |
|--------|-----------|-------------|
| d01 | 132 × 142 (x × y) | Outer domain, coarser resolution |
| d02 | 99 × 99 (x × y) | Inner domain, finer resolution |

### NetCDF Coordinates

- `XLAT` — Latitude (degrees)
- `XLONG` — Longitude (degrees)
- `XTIME` — Time coordinate
- `bottom_top` — Vertical levels (49 levels for 3D variables)
- `south_north` — Y dimension
- `west_east` — X dimension

## Key Variables

### Primary Variables of Interest

| Variable | Description | Units |
|----------|-------------|-------|
| `T2` | 2-meter air temperature | K |
| `loading` | Column-integrated sulfate aerosol loading (in .rds format) | kg/km² |

### Radiation Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `SWDNT`, `SWUPT` | Downward/upward shortwave at TOA (all-sky) | W/m² |
| `SWDNB`, `SWUPB` | Downward/upward shortwave at surface (all-sky) | W/m² |
| `LWDNT`, `LWUPT` | Downward/upward longwave at TOA (all-sky) | W/m² |
| `LWDNB`, `LWUPB` | Downward/upward longwave at surface (all-sky) | W/m² |
| `SWCF`, `LWCF` | Shortwave/longwave cloud forcing | W/m² |
| `OLR` | Outgoing longwave radiation at TOA | W/m² |
| `ALBEDO` | Surface albedo | — |

### Aerosol Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `so4_a01`–`so4_a04` | Sulfate aerosol mass mixing ratio (4 modes) | kg/kg |
| `EXTCOF55` | Aerosol extinction coefficient at 550 nm | m⁻¹ |
| `PM2_5_DRY` | Dry PM2.5 concentration | μg/m³ |
| `PM10` | PM10 concentration | μg/m³ |

### Meteorological Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `T` | Perturbation potential temperature (3D) | K |
| `P`, `PB` | Perturbation and base-state pressure | Pa |
| `PH`, `PHB` | Perturbation and base-state geopotential | m²/s² |
| `U`, `V`, `W` | Wind components | m/s |
| `PBLH` | Planetary boundary layer height | m |
| `LH`, `HFX` | Latent and sensible heat flux | W/m² |
| `QFX` | Surface moisture flux | kg/m²/s |

### Cloud/Moisture Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `QVAPOR` | Water vapor mixing ratio | kg/kg |
| `QCLOUD`, `QICE` | Cloud water/ice mixing ratio | kg/kg |
| `QNDROP` | Cloud droplet number concentration | kg⁻¹ |
| `CLDFRA` | Cloud fraction | — |

## Data Source

Raw WRF-Chem outputs are stored on UCAR Derecho HPC:
`/glade/derecho/scratch/yuhanw/wrfchem/WRF`

Contact: Yuhan Wang (yhanw@stanford.edu) or Yuan Wang (yzwang@stanford.edu)

## Data Loader

The `src/data_loader.py` module provides utilities for loading WRF data:

```python
from src.data_loader import load_variable

# Load single case
da = load_variable('T2', 'd02', 'tmean5d',
                   episode='240527', ensemble='e1', emission_rate='ctl')
# Returns: xarray.DataArray with dims (south_north, west_east)

# Load all ensembles for one episode/rate
da = load_variable('T2', 'd02', 'tmean5d',
                   episode='240527', emission_rate='ctl')
# Returns: xarray.DataArray with dims (ensemble, south_north, west_east)

# Load all available data
da = load_variable('T2', 'd02', 'tmean5d')
# Returns: xarray.DataArray with dims (episode, ensemble, emission_rate, south_north, west_east)
```

Parameters:
- `variable`: Variable name (e.g., `'T2'`, `'SWDNT'`, `'so4_a01'`)
- `domain`: `'d01'` or `'d02'`
- `temporal`: `'allhr'` (hourly) or `'tmean5d'` (5-day mean)
- `episode`: Optional, `'240527'` or `'240727'`
- `ensemble`: Optional, `'e1'`, `'e2'`, or `'e3'`
- `emission_rate`: Optional, `'ctl'`, `'1000'`, `'10000'`, or `'100000'`
