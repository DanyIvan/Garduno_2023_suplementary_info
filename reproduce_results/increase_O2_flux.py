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
description = """In this experiments we use data from Gregory et. al. (2021)
to try to reproduce figure 2 in the manuscript varying temperature""" 

# read input data
O2_flux = np.loadtxt('input_data/gregory_O2_flux.txt')
temp = eval(args.temp)
temp_profile, p0_list = get_temp(temp)
ratio = eval(args.fratio)

# define model generator to increase O2 and CH4 flux for specific temperature
# and relative humidity profiles
def exp_generator(rh):
    def model_generator(idx, model):
        model.vars.t[:] = temp_profile
        model.data.p0 = p0_list
        model.data.relative_humidity = rh
        model.data.use_manabe = 0 
        set_eddy_diff(model, temp=215)
        model.set_surfflux('O2', O2_flux[idx])
        model.set_surfflux('CH4', O2_flux[idx] * ratio)
    return model_generator

# run experiment
nmodels=len(O2_flux)
NAME = f'increase_O2_flux_fratio={args.fratio}_temp={args.temp}_rh={args.rh}'
exp = Experiment('increase_O2_flux', NAME, description, folder_name=args.folder,
    model_generator=exp_generator(eval(args.rh)), nmodels=nmodels)
method = eval(args.method)
exp.run(method=method)