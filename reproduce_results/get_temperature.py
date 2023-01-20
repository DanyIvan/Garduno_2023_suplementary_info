import pandas as pd
import numpy as np

def get_temp(temp, nz=200, dz=0.5e5, cold_stratosphere=False):
    '''Interpolates moist adiabat temperature and pressure profiles to model
    grid'''
    #gas constant
    R = 8.31
    # molar mass of air
    Ma = 28.9635e-3
    # Boltzman constant cgs
    k = 1.38E-16
    grid = np.arange(0, nz * dz, dz)
    # range of allowed temperatures
    allowed_temps = np.arange(240, 405, 5)
    # if temp not in allowed temps raise error
    if temp not in allowed_temps:
        raise(Exception('Temperature must be one of ' + str(allowed_temps)))
    # read data
    else:
        if cold_stratosphere:
            pressure_profile = pd.read_csv('input_data/pressure_profiles_175.csv', 
                names=allowed_temps)
            temp_profile = pd.read_csv('input_data/temperature_profiles_175.csv',
                names=allowed_temps)
        else:
            pressure_profile = pd.read_csv('input_data/pressure_profiles.csv', 
                names=allowed_temps)
            temp_profile = pd.read_csv('input_data/temperature_profiles.csv',
                names=allowed_temps)
        pressure_profile = pressure_profile[temp]
        temp_profile = temp_profile[temp]

    # interpolate to grid
    press = pressure_profile
    alt = 100 * (R * temp_profile * np.log(press[0] / press ))\
        / (Ma * 9.81)  
    temp_interp = np.interp(grid, alt, temp_profile)
    press_interp = np.exp(np.interp(grid, alt, np.log(press)))

    #calculate total number density profile 
    # den = press_interp * 10 / (k * temp_interp)

    # get P0 in bar
    P0 = press_interp[0] * 1e-5
    return list(temp_interp), P0