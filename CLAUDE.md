# Claude Code Style Guide for Regional Climate Analysis

## Project Goal
This project analyzes WRF regional climate model simulations. The data consists of NetCDF files containing various atmospheric variables (temperature, radiation fluxes, aerosols, etc.) from WRF post-processing output.

## Coding Philosophy
This project prioritizes elegant, fail-fast code that surfaces errors quickly rather than hiding them.

### Root Cause Analysis
- Always investigate and understand the root cause of problems before implementing solutions
- Avoid band-aid fixes that mask symptoms without addressing underlying issues
- When unexpected behavior occurs, trace it back to its source rather than applying quick patches
- Document the reasoning behind fixes to prevent similar issues

## Core Style Requirements

### Error Handling
- No input validation on function parameters (except for command-line interfaces)
- No defensive programming - let exceptions bubble up naturally
- Fail fast - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- No try-catch blocks unless absolutely necessary for program logic (not error suppression)
- Assume complete data - do not check for missing data fields. If required data is missing, let the code fail with natural Python errors

### Code Elegance
- Minimize conditional statements - prefer functional approaches, mathematical expressions, and numpy/xarray vectorization
- Favor mathematical clarity over defensive checks
- Use xarray and numpy operations instead of loops and conditionals where possible
- Compute once, use many times - move invariant calculations outside loops and create centralized helper functions
- Use standard Python packages - prefer established methods from scipy, numpy, xarray, etc.

### Code Organization
- All imports at the top of the file - no imports inside functions or scattered throughout the code

### Protected Directories
- Never modify files in `./data/input/` - this directory contains source data that must remain unchanged

### Naming Conventions
- Consistent naming - use the same variable/field names throughout the codebase when referring to the same concept
- Descriptive names preferred - long, clear names are better than short, ambiguous ones

### Function Design
- Functions should assume valid inputs and focus on their core mathematical/logical purpose
- Let Python's natural error messages guide debugging rather than custom error handling

### Output Directories
- Generated figures go in `./figures/`
- Processed data and results go in `./data/output/`
- Excel/CSV exports go in `./data/output/`
