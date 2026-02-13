# WRF-Chem Regional Climate Model Data

This directory contains preprocessed WRF-Chem simulation outputs for analyzing the effects of sulfate aerosol injection on regional climate.

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

## Directory Structure

```
data/input/
├── WRFPOST/                    # Main data directory
│   ├── README.md               # Detailed documentation from data provider
│   ├── geo_em.d01.nc           # Domain 1 geography (land use, terrain, coordinates)
│   ├── geo_em.d02.nc           # Domain 2 geography
│   └── D9_{episode}_{ensemble}_{emission}/   # Case directories
│       ├── allhr_d0{1,2}_{VAR}.nc            # Hourly data (121 timesteps)
│       └── tmean5d_d0{1,2}_{VAR}.nc          # 5-day mean data
└── *.zip                       # Source zip files (delete after extraction)
```

### Directory Naming Convention

```
D9_{episodeID}_{ensembleID}_{emissionRate}[_5x5]
```

Examples:
- `D9_240527_e1_ctl` — May episode, ensemble 1, control run (no injection)
- `D9_240727_e2_10000_5x5` — July episode, ensemble 2, 10 kt/h injection, 5×5 grid

## File Naming Convention

```
{temporal}_{domain}_{VARIABLE}.nc
```

| Component | Values | Description |
|-----------|--------|-------------|
| `temporal` | `allhr`, `tmean5d` | Hourly (121 timesteps over 5 days) or 5-day temporal mean |
| `domain` | `d01`, `d02` | Outer domain (coarser) or inner domain (finer resolution) |
| `VARIABLE` | `T2`, `OLR`, etc. | Variable name in uppercase |

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
