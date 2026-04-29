# Working with PyNGIAB on Anvil HPC

## Build Singularity Image.

1. Log into Anvil HPC https://ondemand.anvil.rcac.purdue.edu using ACCESS account and launch a shell
2. Clone CIROH HPCInfra repo

```
git clone https://github.com/CIROH-UA/NGIAB-HPCInfra.git
```
3. Build singularity image
```
cd NGIAB-HPCInfra/singularity
apptainer build --fakeroot singularity_ngen.sif singularity_ngen.def
```
4. Test singularity image in interactive session (Allocation info: `mybalance`)
```
sinteractive -p shared --nodes=1 --ntasks=1 -A <allocation>
apptainer shell singularity_ngen.sif
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

## Push Singularity Image
0. Prerequisites
    - Quay.io account
    - Profile -> Account Settings -> Robot Accounts -> Create Robot Account
      - `<ROBOT_TOKEN>`
      - `<myrepo+bot>`
      - `<user>`
      - `<repo>`
1. Setup repository access
```
echo "<ROBOT_TOKEN>" | apptainer registry login --username "<myrepo+bot>" --password-stdin docker://quay.io
```
2. Push image to repo
```
apptainer push singularity_ngen.sif oras://quay.io/<user>/<repo>:latest
```