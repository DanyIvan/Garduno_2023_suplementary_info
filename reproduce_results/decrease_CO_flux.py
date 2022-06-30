from PhotochemPy import PhotochemPy
from experiments import Experiment
from get_temperature import get_temp
from set_eddy import set_eddy_diff
import numpy as np
import argparse

# argument parser to pass arguments from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--fratio', help = 'set the methane to oxygen flux ratio',
     default= '0.3')
parser.add_argument('--folder', help='set the folder path where the output will'
    'be saved')
parser.add_argument('--temp', help='set the temperature to run in K',
    default="290")
parser.add_argument('--method', '-m', help='method of integration', default="2")
parser.add_argument('--rh', help='set constant tropospheric relative humidity to run',
     default="0.6")
args = parser.parse_args()

# description for readme.md
description = """In this experiments we reduce the reductant input of the 
    Archean2Proterozoic template to try to cause the GOE""" 

# CO surface flux
CO_flux = np.logspace(11.84, 10, 200) 
temp = eval(args.temp)
temp_profile, p0_list = get_temp(temp)
ratio = eval(args.fratio)

# define model generator to decrease CO for specific temperature
# and relative humidity profiles
def exp_generator(rh):
    def model_generator(idx, model):    
        model.set_surfflux('CO', CO_flux[idx])
        model.set_surfflux('CH4', model.vars.sgflux[model.ispec.index('O2')] * ratio) 
        model.vars.t[:] = temp_profile
        model.data.p0 = p0_list
        model.data.relative_humidity = rh
        model.data.use_manabe = 0 
        set_eddy_diff(model, temp=215)
    return model_generator

# run experiment
nmodels=len(CO_flux)
NAME = f'decreasse_CO_flux_fratio={args.fratio}_temp={args.temp}_rh={args.rh}'
exp = Experiment('decrease_CO_flux', NAME, description, folder_name=args.folder,
    model_generator=exp_generator(eval(args.rh)), nmodels=nmodels)
method = eval(args.method)
exp.run(method=method)