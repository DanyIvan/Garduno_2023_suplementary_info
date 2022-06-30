from PhotochemPy import PhotochemPy
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json


class Experiment:
    '''Class to run sequential experiments with PhotochemPy'''
    def __init__(self,  template, name, description, model_generator, nmodels,
                 nsteps=50000, folder_name=None):
        # folder to save output
        self.output_folder = 'model_output/'
        # experiment name
        self.name = name
        # experiment folder
        new_folder_name = self.output_folder + name + "_" + \
            datetime.today().strftime('%Y%m%d_%H%M%S') + '/'
        self.folder_name = folder_name or new_folder_name
        # description for readme
        self.description = description
        # PhotochemPy templeate to use
        self.template = template
        # generator of Photochempy objects to run
        self.model_generator = model_generator
        # number of models to run
        self.nmodels = nmodels
        # number of time steps to run
        self.nsteps = nsteps

    def run(self, start=0, input_atmosphere=None, method=1, rtol=1e-3,
            atol=1e-27, fast_and_loose=True, out2in=True):
        '''Runs all the models in the generated by the model_generator and saves
            output data to Experiments object folder  
        '''
        Path(self.folder_name).mkdir(exist_ok=True, parents=True)
        self.make_readme()
        atmosphere = input_atmosphere if input_atmosphere else\
            'templates/' + self.template + '/atmosphere.txt'
        model = PhotochemPy('templates/' + self.template +'/species.dat',
                            'templates/' + self.template +'/reactions.rx',
                            'templates/' + self.template +'/settings.yaml',
                            atmosphere,
                            'templates/' + self.template+'/Sun.txt')
        # integration method to use
        met = 'CVODE_BDF' if method == 1 else 'Backward_Euler'
        # run all models
        for idx in range(self.nmodels):
            if idx >= start:
                # create folder to save model output
                model_folder = self.folder_name + '/' + str(idx)
                Path(model_folder).mkdir(exist_ok=True)
                # create and integrate model
                self.model_generator(idx, model)
                model.integrate(nsteps=self.nsteps, method=met, rtol=rtol,
                                atol=atol, fast_and_loose=fast_and_loose)
                save_output(model, model_folder)
                # set output as input for next model
                if out2in:
                    model.out2in()

    def make_readme(self):
        with open(self.folder_name + 'Readme.md', 'w') as f:
            f.write('# Experiment description\n')
            f.write('\n')
            f.write(self.description)

    def set_folder(self, folder):
        self.output_folder = 'experiments_data/' + folder + '/'
        self.folder_name = self.output_folder + self.name + "_" + self.template + "_" + \
            datetime.today().strftime('%Y%m%d_%H%M%S') + '/'
            
def get_atmospheric_columns(model):
    '''Calculates column abundace of all species in a Photochempy object'''
    mixrat_data = pd.DataFrame(model.out_dict())[model.ispec]
    density = model.vars.den
    alt = model.data.z
    def column(species): return np.trapz(species * density, alt)
    atmospheric_columns = mixrat_data.apply(column)
    atmospheric_columns = atmospheric_columns.to_frame().T
    return atmospheric_columns

def get_lifetimes(model):
    '''Calculates lifetimes of all species in a Photochempy object'''
    columns = get_atmospheric_columns(model).values[0][:model.data.nq]   
    flow = model.vars.flow
    lifetimes = np.abs(np.divide(columns, flow, where=flow!=0))
    lifetimes = pd.DataFrame(lifetimes).T 
    lifetimes.columns = model.ispec[:model.data.nq]
    return lifetimes / (60*60*24*365)
    
def get_integrated_rates(model):
    '''Calculates vertically integrated rated of all reactions in a Photochempy
        object'''
    rates_data = pd.DataFrame(model.out_rates())[model.reactions]
    alt = model.data.z
    rates_data = rates_data[[
        col for col in rates_data.columns if col != 'alt']]
    return rates_data.apply(lambda col: np.trapz(col, alt)).to_frame().T

def get_hydrogen_escape(model, species):
    '''Gets hydrogen escape of a Photochempy object'''
    ind = model.ispec.index(species)
    esc = model.out_dict()[species][-1] * \
        model.out_dict()['den'][-1]*model.vars.veff[ind]
    return esc

def save_radiation_data(model, folder):
    '''Saves photon flux to json file'''
    radiation_data = {
        'wavl': (model.data.wavl/10).tolist(),  # wavelenght [nanometers]
        'amean': model.wrk.amean.tolist(),  # mean net flux [?],
        'photon_flux': model.data.flux.tolist()  # [photons/cm^2/s]
    }
    with open(folder + '/radiation_data.json', 'w') as f:
        json.dump(radiation_data, f)

def get_redox_coeffs(model):
    '''Calculates redox fluxes to the atmosphere. This code was taken from
    https://github.com/Nicholaswogan/PhotochemPy/blob/main/src/redox_conservation.f90'''
    oxid_in_new = 0 # oxidized flux input at surface
    oxid_out_new = 0 # oxidized flux output at upper boundary
    red_in_new = 0 # reduced flux input at surface
    red_out_new = 0 # reduced flux output at upper boundary
    red_rain_new = 0 # reduced flux output from rainout
    oxy_rain_new = 0 # oxidized flux output from rainout

    redoxstate = model.data.redoxstate
    for i in range(model.data.nq):
        if redoxstate[i] > 0:
            oxid_in_new += model.vars.flow[i]*redoxstate[i]
            oxid_out_new += model.vars.fup[i]*redoxstate[i]
            oxy_rain_new += model.vars.sr[i]*redoxstate[i]
        elif redoxstate[i] < 0:
            red_in_new += model.vars.flow[i]*redoxstate[i]*(-1.0)
            red_out_new += model.vars.fup[i]*redoxstate[i]*(-1.0)
            red_rain_new += model.vars.sr[i]*redoxstate[i]*(-1.0)

    # distributed fluxes
    for i in range(model.data.nq):
        if model.vars.lbound[i] == 3:
            if redoxstate[i] > 0:
                oxid_in_new += model.vars.distflux[i]*redoxstate[i]
            elif redoxstate[i] < 0:
                red_in_new += model.vars.distflux[i]*redoxstate[i]*(-1)

    redox_new = red_in_new - red_out_new \
        - oxid_in_new + oxid_out_new \
        - red_rain_new + oxy_rain_new

    redox_arr = [red_in_new, red_out_new, oxid_in_new,
                 oxid_out_new, red_rain_new, oxy_rain_new]

    # redox_new should be small compared to largest redox fluxes
    redox_factor = redox_new/np.max(np.abs(redox_arr))
    redox_coeffs = {'oxid_in_new': oxid_in_new,
                    'oxid_out_new': oxid_out_new,
                    'red_in_new': red_in_new,
                    'red_out_new': red_out_new,
                    'red_rain_new': red_rain_new,
                    'oxy_rain_new': oxy_rain_new,
                    'redox_new': redox_new,
                    'redox_factor': redox_factor}
    return pd.DataFrame({k: [v] for k, v in redox_coeffs.items()})

def get_redox_and_koxy_components(model):
    '''Calculates redox fluxes associated with each species in the atmosphere to
    calculate the redox factor and the koxy factor of the model.
    The sum of all fluxes for all species must be equal to the output of 
    get_redox_coeffs'''
    data_redox = pd.DataFrame(
        {'flow': model.vars.flow,
        'fup': model.vars.fup,
        'sr': model.vars.sr,
        'distflux': model.vars.distflux,
        'redox_state': model.data.redoxstate[:model.data.nq],
        'species': model.ispec[:model.data.nq]
        }) 
    redoxstate = model.data.redoxstate[:model.data.nq]
    positive_redoxstate = np.array([x if x > 0 else 0  for x in redoxstate])
    negative_redoxstate = np.array([x if x < 0 else 0  for x in redoxstate])

    data_redox['oxid_in'] = data_redox.flow * positive_redoxstate +\
        data_redox.distflux * positive_redoxstate 
    data_redox['oxid_out'] = data_redox.fup * positive_redoxstate
    data_redox['oxy_rain'] = data_redox.sr * positive_redoxstate
    data_redox['red_in'] = data_redox.flow * negative_redoxstate * (-1.0) +\
        data_redox.distflux * negative_redoxstate * (-1.0)
    data_redox['red_out'] = data_redox.fup * negative_redoxstate * (-1.0)
    data_redox['red_rain'] = data_redox.sr * negative_redoxstate * (-1.0) 

    # check values
    # this = data_redox[['oxid_in', 'oxid_out',  'red_in', 'red_out', 'red_rain',
    # 'oxy_rain']].sum().values
    # other = get_redox_coeffs(model)
    # if sum(this == other.iloc[0,:6].values) != 6:
    #     raise Exception("Values do not coincide")

    # koxy components
    data_koxy = pd.DataFrame(
        {'flow': model.vars.flow,
        'escape': model.vars.veff * model.vars.den[-1] * model.vars.usol_out[:, -1],
        'redox_state': model.data.redoxstate[:model.data.nq],
        'species': model.ispec[:model.data.nq]
        }) 
    data_koxy['flow_red'] = data_koxy.flow * -negative_redoxstate
    data_koxy['escape_red'] = data_koxy.escape * -negative_redoxstate
    data_koxy['flow_oxi'] = data_koxy.flow * positive_redoxstate
    data_koxy['escape_oxi'] = data_koxy.escape * positive_redoxstate
    data_koxy['Fred'] = data_koxy.flow_red - data_koxy.escape_red
    data_koxy['Foxi'] = data_koxy.flow_oxi - data_koxy.escape_oxi

    # check values
    # sumed = data_koxy.sum()
    # if (sumed.Foxi/sumed.Fred != model.koxy()):
    #     raise Exception("Values do not coincide")
    return data_redox, data_koxy
    


def save_output(model, model_folder):
    # mixing ratio profiles
    mix_ratios = pd.DataFrame(model.out_dict())
    mix_ratios['H2O_sat_mixrat'] = model.vars.h2osat
    mix_ratios['relh'] = mix_ratios['H2O'] / mix_ratios['H2O_sat_mixrat']
    mix_ratios['H2O_sat_press'] = mix_ratios['H2O_sat_mixrat'] * model.vars.p
    # colum abundances
    atmospheric_columns = get_atmospheric_columns(model)
    atmospheric_columns['H2esc'] = get_hydrogen_escape(model, 'H2')
    atmospheric_columns['Hesc'] = get_hydrogen_escape(model, 'H')
    atmospheric_columns['koxy'] = model.koxy()
    atmospheric_columns['redox_factor'] = model.redox_factor
    # reaction rates
    rates = pd.DataFrame(model.out_rates())
    # not sure if this is the rainout
    rainout = pd.DataFrame({k: [v] for k, v in zip(model.ispec[:model.data.nq],
        model.vars.sr)})
    # redox fluxes
    redox_coeffs = get_redox_coeffs(model)
    # integrated reaction rates
    integrated_rates = get_integrated_rates(model)
    # photon flux
    radiative_flux = get_radiative_flux(model)
    # redox fluxes for each species
    redox_components, koxy_components = get_redox_and_koxy_components(model)
    # lifetimes
    lifetimes = get_lifetimes(model)
    # save output to csv files
    O2_flux = model.vars.sgflux[model.ispec.index('O2')]
    CO_flux = model.vars.sgflux[model.ispec.index('CO')]
    for df in [mix_ratios, rates, atmospheric_columns,
        integrated_rates, rainout, redox_coeffs, radiative_flux, 
        redox_components, koxy_components, lifetimes]:
        df['O2_flux_lb'] = O2_flux
        df['CO_flux_lb'] = CO_flux
        df['temp_lb'] = model.vars.t[0]
        df['relh_lb'] = model.data.relative_humidity
        df['converged'] = model.converged
    files = ['mix_ratios', 'rates', 'atmospheric_columns',
        'integrated_rates', 'rainout', 'redox_coeffs', 'radiative_flux',
        'redox_components', 'koxy_components', 'lifetimes']
    for i, df in enumerate([mix_ratios, rates, atmospheric_columns,
        integrated_rates, rainout, redox_coeffs, radiative_flux, 
        redox_components, koxy_components, lifetimes]):
        df.to_csv(model_folder + '/' + files[i] + '.csv', index=False)
    # save output atmosphere.txt file
    model.out2atmosphere_txt(model_folder + '/atmosphere.txt')
    save_radiation_data(model, model_folder)

def photons_to_watts(pflux, wavl):
    '''Convert photon flux to watts'''
    c = 3e8  # speed of light m/s
    h = 6.62607004e-34  # plank's constant Js
    watts = (pflux * c * h * 1e3) / (wavl * 1e-9)
    return watts


def is_eval(x):
    '''cheks if eval can be applied to x'''
    try:
        eval(x)
        return True
    except:
        return False


def get_radiative_flux(model, watts=False):
    '''Calculates photon flux through the atmsophere'''
    radiative_flux = model.wrk.amean * model.data.flux
    radiative_flux = pd.DataFrame(radiative_flux).T
    radiative_flux.columns = model.data.z/1e5
    if watts:
        radiative_flux = radiative_flux.apply(lambda col: photons_to_watts(col,
            model.data.wavl[:-1]/10))
    radiative_flux['wavl'] = model.data.wavl[:-1]/10
    return radiative_flux
