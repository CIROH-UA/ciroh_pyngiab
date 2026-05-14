# Working with PyNGIAB on University of Alabama Panterhei HPC

## Build Singularity Image.

1. Follow instructions https://hub.ciroh.org/docs/services/on-prem/Pantarhei/access to get access to the HPC system
2. Clone CIROH HPCInfra repo

```
git clone https://github.com/CIROH-UA/NGIAB-HPCInfra.git
```
3. Build singularity image
`singularity` is available on login node. However, building the original `.def` file results in error specifically for the step `dnf update -y`

  3.1. Edit `singularity_ngen.def` to comment `dnf update -y` line 

```
cd NGIAB-HPCInfra/singularity
# comment `dnf update -y` in `singularity_ngen.def`
apptainer build --fakeroot singularity_ngen.sif singularity_ngen.def
```
4. Test singularity image in interactive session (Allocation info: `mybalance`)
```
srun --ntasks=1 --cpus-per-task=2 --time=00:15:00 --pty bash
singularity shell singularity_ngen.sif
/dmod/bin/ngen
```
Expected output simialar to following
```
NGen Framework 0.3.0
Usage: 
/dmod/bin/ngen <catchment_data_path> <catchment subset ids> <nexus_data_path> <nexus subset ids> <realization_config_path>
Arguments for <catchment subset ids> and <nexus subset ids> must be given.
Use "all" as explicit argument when no subset is needed.
Build Info:
  NGen version: 0.3.0
  Parallel build
  NetCDF lumped forcing enabled
  Fortran BMI enabled
  C BMI enabled
  Python active
    Embedded interpreter version: 3.9.25
  Routing active
Python Environment Info:
  VIRTUAL_ENV environment variable: (not set)
  Discovered venv: None
  System paths:
    
    /usr/lib64/python39.zip
    /usr/lib64/python3.9
    /usr/lib64/python3.9/lib-dynload
    /ngen/.venv/lib64/python3.9/site-packages
    /ngen/.venv/lib/python3.9/site-packages
```