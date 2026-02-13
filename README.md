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

The loader transparently handles `.rds` files (R data format) when `.nc` files are not available. This allows variables like `loading` that are stored only in `.rds` format to be loaded using the same interface.

### Grid Cell Areas

Grid cell areas vary across the domain due to map projection distortion. The `load_cell_area` function computes cell areas using the WRF map scale factor method:

```
area = (DX × DY) / (MAPFAC_M²)
```

Where:
- `DX`, `DY`: Nominal grid spacing from file attributes (d01: 27 km, d02: 9 km)
- `MAPFAC_M`: Map scale factor at mass points from geo_em files

```python
from src.data_loader import load_cell_area

area = load_cell_area('d02')  # Returns xarray DataArray in km²
# Resulting area ranges: d01: 528–844 km², d02: 67–77 km²
```

## Ratio Analysis

The `src/ratio_analysis.py` module computes gridded ratio fields with error propagation, showing how each grid cell contributes to the total domain-wide change.

### Running the Analysis

```bash
python -m src.ratio_analysis
```

### Methodology

1. **Time-average**: Hourly data (121 timesteps) averaged to single 2D fields
2. **Ensemble statistics**: Mean and standard error computed across 3 ensemble members
3. **Differences from control**: `d = mean_rate - mean_ctl` with propagated SE
4. **Area-weighted sums**: `S = Σ(d × area)` over all grid cells
5. **Ratio fields**: `r = d / S` (units: 1/km²) — contribution per unit area
6. **Average across rates**: Mean of ratios from 1000, 10000, 100000 t/h injection rates
7. **Sorted tables**: Grid cells sorted by ratio (positive → negative), with cumulative sums

### Output Files

Excel files (4 total, one per variable × episode):
```
data/output/
├── T2_ratio_analysis_240527.xlsx
├── T2_ratio_analysis_240727.xlsx
├── loading_ratio_analysis_240527.xlsx
└── loading_ratio_analysis_240727.xlsx
```

Each Excel file contains 18,744 rows (one per grid cell) with columns:
- `ratio`: Mean ratio across emission rates (1/km²)
- `cell_area_km2`: Grid cell area (km²)
- `cumulative_contribution`: Running sum of ratio × area (dimensionless, sums to 1.0)
- `cumulative_area_km2`: Running sum of cell areas (km²)

PDF figures (4 total):
```
figures/
├── T2_ratio_analysis_240527.pdf
├── T2_ratio_analysis_240727.pdf
├── loading_ratio_analysis_240527.pdf
└── loading_ratio_analysis_240727.pdf
```

Each PDF contains two plots:
1. **Ratio vs Cumulative Area**: Shows how ratio values vary across sorted grid cells
2. **Cumulative Contribution vs Cumulative Area**: Shows what fraction of total change is explained by the top N km² of grid cells

### Error Propagation

Standard errors are propagated through each calculation step:
- **Ensemble SE**: `SE = σ / √3`
- **Difference SE**: `SE_d = √(SE_rate² + SE_ctl²)`
- **Sum SE**: `SE_S = √(Σ(SE_d² × area²))`
- **Ratio SE**: `SE_r = SE_d / |S|`
- **Mean Ratio SE**: `SE_r_mean = √(SE_r1000² + SE_r10000² + SE_r100000²) / 3`

### Units Summary

| Quantity | T2 | loading |
|----------|-----|---------|
| Raw data | K | kg/km² |
| Difference (d) | K | kg/km² |
| Area-weighted sum (S) | K km² | kg |
| Ratio (r = d/S) | 1/km² | 1/km² |
