README – Preprocessed WRF-Chem Outputs

This directory contains preprocessed WRF-Chem outputs prepared for collaboration and downstream analysis.

====================================================================
1. Folder Structure and Case Naming
====================================================================

The parent directory contains 24 subfolders. Each subfolder corresponds to one WRF-Chem simulation case.

Case breakdown:
- 2 episodes
- 3 ensembles per episode
- 3 injection experiments + 1 control run per ensemble

Total number of cases:
2 × 3 × (3 + 1) = 24

Naming convention for each case folder:

D9_{episodeID}_{ensembleID}_{expID_or_ctl}

Examples:

D9_240727_e2_1000_5x5
  - Episode: 240727 (July episode)
  - Ensemble: e2
  - Injection experiment: 1000 t h-1 (1 kt h-1)
  - Injection region: 5 × 5 grid cells in d02

D9_240527_e3_ctl
  - Episode: 240527 (May episode)
  - Ensemble: e3
  - Control run (no injection)

====================================================================
2. Files Inside Each Case Folder
====================================================================

Each case folder contains preprocessed NetCDF files with the following naming conventions:

allhr_{domain}_{varID}.nc
tmean5d_{domain}_{varID}.nc

File types:

(1) allhr_*.nc
    - Hourly-resolved outputs
    - 121 time steps (including initial time)
    - Start time: 00:00 local time (after 2-day spin-up)
    - Duration: 5 days

(2) tmean5d_*.nc
    - 5-day mean fields
    - Generated to reduce file size

Note: Daytime-mean files (e.g., 06:00–18:00 local time averages) are not explicitly provided due to limited time. They can be computed by subsetting the corresponding allhr files.

====================================================================
3. Domains and Grid Dimensions
====================================================================

Domain d01: 132 (x) × 142 (y)
Domain d02: 99 (x) × 99 (y)

All variables follow the native WRF-Chem grid for the respective domain.

Grid geography can be found in "geom_em.d01.nc" and "geom_em.d02.nc"

====================================================================
4. Key Variables of Interest
====================================================================

While many variables are included, the two most relevant for current analyses are:

T2
  - 2 m air temperature (K)

loading
  - Column-integrated sulfate aerosol loading (kg/km2)
  - Derived variable, so stored in R format (.rds), not NetCDF
  - Same spatial and temporal dimensions as other gridded variables in NetCDF

====================================================================
5. List of Other Processed Variables and Definitions
====================================================================

Meteorology and dynamics:
U, V        - Zonal and meridional wind components (m s-1)
W           - Vertical velocity (m s-1)
PBLH        - Planetary boundary layer height (m)
T           - Perturbation potential temperature (K)
P, PB       - Perturbation and base-state pressure (Pa)
PH, PHB     - Perturbation and base-state geopotential (m2 s-2)
T2          - 2 m air temperature (K)

Moisture and clouds:
QVAPOR      - Water vapor mixing ratio (kg kg-1)
QCLOUD      - Cloud water mixing ratio (kg kg-1)
QICE        - Cloud ice mixing ratio (kg kg-1)
QNDROP      - Cloud droplet number concentration (kg-1)
CLDFRA      - Cloud fraction

Aerosols and chemistry:
so4_a01, so4_a02, so4_a03, so4_a04
            - Sulfate aerosol mass mixing ratios in different aerosol modes (kg kg-1)
EXTCOF55    - Aerosol extinction coefficient at 550 nm (m-1)
PM2_5_DRY   - Dry PM2.5 mass concentration (µg m-3)
PM10        - PM10 mass concentration (µg m-3)
loading     - Column-integrated sulfate aerosol loading (derived)

Radiation and surface fluxes:
OLR         - Outgoing longwave radiation at top of atmosphere (W m-2)
ALBEDO      - Surface albedo

Shortwave radiation:
SWDNTC / SWUPTC   - Downward/upward shortwave flux at TOA (clear-sky)
SWDNBC / SWUPBC   - Downward/upward shortwave flux at surface (clear-sky)
SWDNT / SWUPT     - Downward/upward shortwave flux at TOA (all-sky)
SWDNB / SWUPB     - Downward/upward shortwave flux at surface (all-sky)

Longwave radiation:
LWDNTC / LWUPTC   - Downward/upward longwave flux at TOA (clear-sky)
LWDNBC / LWUPBC   - Downward/upward longwave flux at surface (clear-sky)
LWDNT / LWUPT     - Downward/upward longwave flux at TOA (all-sky)
LWDNB / LWUPB     - Downward/upward longwave flux at surface (all-sky)

Surface energy and moisture fluxes:
LH          - Latent heat flux (W m-2)
HFX         - Sensible heat flux (W m-2)
QFX         - Surface moisture flux (kg m-2 s-1)

====================================================================
6. Notes
====================================================================

Raw WRF-Chem outputs are stored on UCAR Derecho HPC under path:
/glade/derecho/scratch/yuhanw/wrfchem/WRF

For any questions or access to raw data, please contact Yuhan Wang (yhanw@stanford.edu) or Yuan Wang (yzwang@stanford.edu)
