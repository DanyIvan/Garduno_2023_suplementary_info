# Reproduce our results

## Install PhotochemPy
To reproduce our results, you first need to install the version of
PhotochemPy that we use. We made minor changes to the [original code](https://github.com/Nicholaswogan/PhotochemPy/)
to run our simulations. Clone this version with the following command:

```
git clone -b experimental_daniel https://github.com/DanyIvan/PhotochemPy
```

Then follow the PhotochemPy installation instructions:

- Create a `conda` environment with all dependencies:
  
```
conda create -n photochempy -c conda-forge python numpy scipy scikit-build
```

- Navigate to the root directory with a terminal, activate your new conda environment, then install with setup.py:
  
```
conda activate photochempy
python setup.py install
```

## Install extra dependencies

Some additional packages specified in the file
`requirements.txt` are needed to run our python scripts. You can install these packages in the photochempy `conda`
environment with:

```
conda install --file requirements.txt
```

## Run scripts

The `reproduce_results/` folder contains the python scripts to reproduce our result.
The `reproduce_results/templates/` folder contains the PhotochemPy templates we use.
The `reproduce_results/input_data/` folder contains the oxygen fluxes, temperature profiles,
and relative humidity profiles we use.

To reproduce our results, activate the photochempy `conda` environment generated 
in the previous steps. Then navigate to the `reproduce_results/` folder and run the
`increase_O2_flux.py` or the `decrease_CO_flux.py` scripts. You can specify the
surface temperature, surface relative humidity, and methane to oxygen flux
ratio for each run from the terminal. For example:

```
python increase_O2_flux.py --fratio 0.3 --rh 0.6 --temp 270
```

In this case, you will reproduce our results for the case in which the oxygen flux increases, with a methane to oxygen flux ratio of 0.3, a surface temperature of 270, and a tropospheric relative humidity of 0.6. The model output from this run will be saved in the `model_output/` folder.

# Remake our plots

The `plots/` folder contains a python script `make_plots.py` to remake all the plots we show in the manuscript and some other plots. The `plots/model_output/` folder contains all the model output necessary to make these plots.

To remake our plots, navigate to the `plots/` folder, activate the photochempy `conda` environment generated in the previous steps, and run:

```
python make_plots.py
```

# Boundary Conditions

The folder `boundary_conditions/` contains the species boundary conditions machine-readable files that we use in the `species.dat` files. We keep these boundary conditions constant for all species across all runs, except for methane, oxygen, and carbon monoxide. We specify each run's surface flux boundary conditions in CSV files for methane, oxygen, and carbon monoxide. The fluxes units are photochemical units ($1pu = 1 molecule/cm^{2}/s$).

# Reaction rates

The `reaction_rates/` folder contains the machine-readable reaction rates file that we use. We use the reaction rates of the PhotochemPy `Archean2Proterozoic` templates. You can also view these rates here: 
    
<https://github.com/DanyIvan/PhotochemPy/blob/main/input/templates/Archean2Proterozoic/reactions.rx>