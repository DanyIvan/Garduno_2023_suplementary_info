import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from plot_helpers import add_colorbar, scatterplot, lineplot,  print_name,\
    filter_data, format_minor_ticks, sci_notation, format_minor_ticks,\
    subplots_centered, add_horizontal_legend

# size of figure to fit in a4 page size
a4_x, a4_y = (8.27, 11.69)

# set ticks size
mpl.rc('xtick', labelsize=6) 
mpl.rc('ytick', labelsize=6)

flux_labels = {'increase_O2': r'Surface $O_2$ flux [$/cm^2/s$]',
    'decrease_CO': r'Surface $CO$ flux [$/cm^2/s$]'}

@print_name
def boundary_conditions(scenario='increase_O2'):
    '''Plots boundary conditions for temperature, eddy diffusivity, pressure and
    relative humidity in the atmosphere ''' 
    if scenario == 'increase_O2':
        bcs_data = 'model_output/increase_O2_flux_case=0.3_bcs.csv'
    elif scenario == 'colder_strat':
        bcs_data = 'model_output/increase_O2_flux_case=0.3_colder_stratosphere_1_bcs.csv'
    elif scenario == 'eddy_sensitivity':
        bcs_data = 'model_output/increase_O2_flux_case=0.3_eddy_sensitivity_bcs.csv'
    else:
        bcs_data = 'model_output/decrease_CO_flux_case=0.3_bcs.csv'
    data = pd.read_csv(bcs_data, compression='gzip')
    temps = np.arange(250, 370, 10.)
    data = data[data.temp_lb.isin(temps)]
    datarh6 = data[data.relh_lb==0.6]

    fig, axes = subplots_centered(3,2,(a4_x * 0.75, a4_y * 0.6), 5)
    # temperature
    plt.sca(axes[0])
    lineplot(datarh6, 'T', 'alt', 'temp_lb', lw=0.7)
    axes[0].set_ylabel('Altitude [Km]', fontsize=8)
    axes[0].set_xlabel('Temperature [K]', fontsize=8)
    axes[0].text(0.1, 0.9, 'a)', transform=axes[0].transAxes, size=8)
    # eddy diffusivity
    plt.sca(axes[1])
    lineplot(datarh6, 'edd', 'alt', 'temp_lb', lw=0.7)
    axes[1].set_xlabel(r'Eddy diffusivity [$cm^{2} s^{-1}$]',fontsize=8)
    plt.xscale('log')
    axes[1].text(0.1, 0.9, 'b)', transform=axes[1].transAxes, size=8)
    # pressure
    plt.sca(axes[2])
    lineplot(datarh6, 'press', 'alt', 'temp_lb', lw=0.7)
    axes[2].set_ylabel('Altitude [Km]', fontsize=8)
    axes[2].set_xlabel('Pressure [Bar]', fontsize=8)
    axes[2].text(0.9, 0.9, 'c)', transform=axes[2].transAxes, size=8)
    plt.xscale('log')
    # relative humidity
    plt.sca(axes[3])
    for rh in data.relh_lb.unique():
        data_rh = data[data.relh_lb == rh]
        lineplot(data_rh, 'relh', 'alt', 'temp_lb', lw=0.7)
    axes[3].set_xlabel(r'Relative humidity', fontsize=8)
    axes[3].text(0.9, 0.9, 'd)', transform=axes[3].transAxes, size=8)

    plt.sca(axes[4])
    data_surface = data[data.alt==0.25]
    scatterplot(data_surface, 'temp_lb', 'H2O', 'relh_lb', s=2, palette='viridis')
    lineplot(data_surface, 'temp_lb', 'H2O', 'relh_lb', lw=0.7, palette='viridis',
        labels=[0.2, 0.6, 1.0])
    legend = plt.legend(fontsize=6, loc='lower right', frameon=False,
        title='Relative \nHumidity')
    plt.setp(legend.get_title(),fontsize='xx-small')
    plt.yscale('log')
    axes[4].set_ylabel(r'Surface $H_2O$' + '\n mixing ratio', fontsize=8)
    axes[4].set_xlabel(r'Surface Temperature [K]', fontsize=8)
    axes[4].text(0.1, 0.9, 'e)', transform=axes[4].transAxes, size=8)

    plt.subplots_adjust(wspace=0.4, hspace=0.35)
    add_colorbar(fig, axes, temps, 'Surface Temperature [K]', pad=0.08,
        ticks=temps)
    if scenario == 'increase_O2':
        fig.savefig('atm_bcs.pdf', bbox_inches='tight')
    elif scenario == 'colder_strat':
        fig.savefig('atm_bcs_cold_strat.pdf', bbox_inches='tight')
    elif scenario == 'eddy_sensitivity':
        fig.savefig('atm_bcs_eddy_sensitivity.pdf', bbox_inches='tight')
    plt.close('all')

@print_name
def gregory_case1_vs_this_study():
    '''Plots a comparison of Grgory et. al. (2021) results and our results'''
    data = pd.read_csv('model_output/gregory_vs_this_study_mixrat.csv',
        compression='gzip')
    data = data[data.alt == 0.25]
    data = data[data.converged]
    species = data.columns[:-5]
    formulas = {'O2': r'$O_2$', 'CH4':r'$CH_4$', 'O3':r'$O_3$', 'OH':r'$OH$', 
        'H2':r'$H_2$', 'CO': r'$CO$', 'H2O':r'$H_{2}O$'}
    palette = sns.color_palette('coolwarm', 11)

    # make handles for legend
    colors = [palette[0], palette[-1]]
    shapes = ["o", "^", "*"]
    
    def make_handles(m,c, ls='none', alpha=None):
        return plt.plot([],[],marker=m, color=c, ls=ls, alpha=alpha)[0]
    handles = [make_handles("s","w", alpha=0)] + [make_handles("s", colors[i]) for i in
        range(2)] + [make_handles("s","w", alpha=0)] + [make_handles(shapes[i], "k") 
        for i in range(3)]

    labels =["Simulations","Gregory et. al. (2021)", "This study",
        r"$CH_4$:$O_2$ flux ratio", "0.094", "0.3", "0.45"]

    cases = [0.094, 0.3, 0.45]
    for sp in species:
        fig = plt.figure(figsize=(a4_x*0.6, a4_y*0.25))
        if sp == 'O2':
            plt.axhline(0.21, color='k', linestyle='--', lw=0.7)
        for idx_color, data_source in enumerate(data.data_source.unique()):
            for idx_shape, case in enumerate(cases):
                d = data[(data.data_source == data_source) & (data.case == case)]
                plt.scatter(d.O2_flux_lb, d[sp], color=colors[idx_color],
                    marker=shapes[idx_shape], s=6)
        plt.xlabel(r'Surface $O_2$ flux [$/cm^2/s$]', fontsize=8)
        plt.ylabel(formulas[sp] + ' surface mixing ratio',fontsize=8)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(handles, labels, fontsize=6, frameon=False)
        fig.savefig('gregory_vs_this_study_' + sp + '.pdf', bbox_inches='tight')

    # O2 plot
    labels_O2 =["Simulations","Gregory et. al. (2021)", "This study",
        r"$CH_4$:$O_2$ flux ratio", "0.094", "0.3", "0.45", 
        "Proxy constraints", "PAL", "Before GOE", "After GOE"]
    f1 = lambda c: plt.plot([],[], linestyle='--', color=c)[0]
    handles_O2 = handles + [make_handles("s","w", alpha=0)] +\
        [make_handles(None, 'k', ls='--')] +\
        [make_handles("s","#D3D3D3"),  make_handles("s","#ADD8E6")]
    fig = plt.figure(figsize=(a4_x*0.6, a4_y*0.25))
    plt.axhline(0.21, color='k', linestyle='--', lw=0.7, zorder=20)
    for idx_color, data_source in enumerate(data.data_source.unique()):
        for idx_shape, case in enumerate(cases):
            d = data[(data.data_source == data_source) & (data.case == case)]
            plt.scatter(d.O2_flux_lb, d['O2'], color=colors[idx_color],
                marker=shapes[idx_shape], s=6, zorder=10)
    plt.gca().axhspan(1e-11, 0.21e-6, color='#D3D3D3', alpha=0.5, zorder=0)
    plt.ylim([1e-11, 2])
    plt.gca().axhspan(0.21e-3, 0.21e-1, color='#ADD8E6', alpha=0.5, zorder=0)
    plt.xlabel(r'Surface $O_2$ flux [$/cm^2/s$]', fontsize=8)
    plt.ylabel(formulas['O2'] + ' surface mixing ratio',fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(handles_O2, labels_O2, fontsize=6, frameon=False)
    fig.savefig('gregory_vs_this_study_O2_with_proxies.pdf', bbox_inches='tight')

    # O3 Colulm
    data_O3 = pd.read_csv('model_output/gregory_vs_this_study_O3_column.csv', compression='gzip')
    data_O3 = data_O3[data_O3.converged]
    fig = plt.figure(figsize=(a4_x*0.6, a4_y*0.25))
    for idx_color, data_source in enumerate(data_O3.data_source.unique()):
        for idx_shape, case in enumerate(cases):
            d = data_O3[(data_O3.data_source == data_source) & (data_O3.case == case)]
            plt.scatter(d.O2_flux_lb, d['O3_column'], color=colors[idx_color],
                marker=shapes[idx_shape], s=6)
    plt.xlabel(r'Surface $O_2$ flux [$/cm^2/s$]', fontsize=8)
    plt.ylabel(r'$O_3$ column', fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(handles, labels, fontsize=6, frameon=False)
    fig.savefig('gregory_vs_this_study_O3_column.pdf', bbox_inches='tight')
    plt.close('all')

@print_name
def OH_main_sources(scenario='increase_O2'):
    '''
    Plots the main reactions that produce OH
    ''' 
    # TODO
    if scenario == 'increase_O2':
        data_int_rates = 'model_output/increase_O2_flux_case=0.3_some_int_rates.csv'
        flux = 'O2_flux_lb'
    else:
        data_int_rates = 'model_output/decrease_CO_flux_case=0.3_some_int_rates.csv'
        flux = 'CO_flux_lb'
    # read and filter data
    data_int_rates = pd.read_csv(data_int_rates, compression='gzip')
    data_int_rates = filter_data(data_int_rates, scenario=scenario)
    data_360 = data_int_rates[data_int_rates.temp_lb == 360]
    temps = np.arange(250, 370, 10.)
    last_color = lambda palette:sns.color_palette(palette, len(temps))[-1]

    fig, axes = plt.subplots(1,1, figsize=(a4_x*0.3, a4_y*0.25))
    scatterplot(data_int_rates, flux, 'H2O + O1D = OH + OH', 'temp_lb',
        s=2, palette="light:#cf453c")
    plt.scatter(data_360[flux], data_360['H2O + O1D = OH + OH'], s=2,
        color=last_color("light:#cf453c"), label=r'$H_{2}O + O(^1D) \rightarrow 2OH$')
    r'$H_{2}O + hv \rightarrow H + OH$'
    scatterplot(data_int_rates, flux, 'H2O + HV = H + OH', 'temp_lb',
        s=2, palette="light:#516ddb")
    plt.scatter(data_360[flux], data_360['H2O + HV = H + OH'], s=2,
        color=last_color("light:#516ddb"), label=r'$H_{2}O + hv \rightarrow H + OH$')
    plt.ylabel(r'Integrated rate [$/cm^2/s$]', fontsize=8)
    plt.xlabel(flux_labels[scenario], fontsize=8)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=6, frameon=False)

    if scenario == 'increase_O2':
        plt.savefig('increase_O2_OH_main_sources.pdf', bbox_inches='tight', transparent=True)
    else:
        plt.savefig('decrease_CO_OH_main_sources.pdf', bbox_inches='tight', transparent=True)

@print_name
def O3_and_H2O_after_GOE(scenario = 'increase_O2'):
    '''
    Plots the profiles of O3 and H2O at different temperatures
    ''' 
    # TODO
    if scenario == 'increase_O2':
        data_rates = 'model_output/increase_O2_flux_case=0.3_some_species_profiles.csv'
    else:
        data_rates = 'model_output/decrease_CO_flux_case=0.3_some_species_profiles.csv'

    # read and filter data
    data = pd.read_csv(data_rates, compression='gzip')
    data = data[data.converged]
    fluxes = data.O2_flux_lb.unique()
    data = data[data.relh_lb == 0.6]
    data = data[data.O2_flux_lb == fluxes[-1]]
    temps = np.arange(250, 370, 20.)

    data_360 = data[data.temp_lb == 360]
    last_color = lambda palette:sns.color_palette(palette, len(temps))[-1]

    fig, axes = plt.subplots(1,1, figsize=(a4_x*0.3, a4_y*0.25))
    lineplot(data, 'O3', 'alt', 'temp_lb', lw=0.9, palette="light:#cf453c", alpha=0.8)
    plt.plot(data_360.O3, data_360.alt, label=r'$O_3$', color=last_color("light:#cf453c"),
        lw=0.9, alpha=0.8)
    lineplot(data, 'H2O', 'alt', 'temp_lb', lw=0.9, palette="light:#516ddb", alpha=0.8)
    plt.plot(data_360.H2O, data_360.alt, label=r'$H_2O$', color=last_color("light:#516ddb"),
        lw=0.9, alpha=0.8)
    lineplot(data, 'OH', 'alt', 'temp_lb', lw=0.9, palette="light:k", alpha=0.8)
    plt.plot(data_360.OH, data_360.alt, label=r'$OH$', color=last_color("light:k"),
        lw=0.9, alpha=0.8)
    # lineplot(data, 'OH + O3 = HO2 + O2', 'alt', 'temp_lb', lw=0.9, palette='flare')
    plt.xscale('log')
    plt.ylabel('Altitude [km]', fontsize=8)
    plt.xlabel('Mixing ratio\n' +r'Surface $O_2$ flux:' + sci_notation(fluxes[-1]) +\
        r'$/cm^2/s$' , fontsize=8)
    plt.legend(fontsize=6, frameon=False)

    if scenario == 'increase_O2':
        plt.savefig('increase_O2_O3_and_H2O.pdf', bbox_inches='tight', transparent=True)
    else:
        plt.savefig('decrease_CO_O3_and_H2O.pdf', bbox_inches='tight', transparent=True)

@print_name
def OH_main_sources_O3_and_H2O_after_GOE_colorbars():
    '''
    Plots colorbars
    ''' 
    temps = np.arange(250, 370, 10)
    fig, axes = plt.subplots(1,1, figsize=(a4_x*0.5, a4_y*0.25))
    cmap1 =  sns.color_palette("light:#cf453c", len(temps), as_cmap=True)
    cmap2 =  sns.color_palette("light:#516ddb", len(temps), as_cmap=True)
    cmap3 =  sns.color_palette("light:k", len(temps), as_cmap=True)
    norm = mpl.colors.Normalize(vmin=min(temps), vmax=max(temps))

    cb3 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap3), 
        ax=axes, orientation='horizontal', aspect=40)
    cb3.set_ticks(temps)
    cb3.set_label(label='Surface Temperature [K]', size=8)
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap2), 
        ax=axes, orientation='horizontal', aspect=40)
    cb2.set_ticks([])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap1), 
        ax=axes, orientation='horizontal', aspect=40)
    cb1.set_ticks([])
    plt.savefig('colorbars.pdf', bbox_inches='tight', transparent=True)

@print_name
def H2O_mixrat_vs_temp(scenario='increase_O2'):
    '''Plots H2O surface mixing ratio vs surface temperature'''
    if scenario == 'increase_O2':
        bcs_data = 'model_output/increase_O2_flux_case=0.3_bcs.csv'
    else:
        bcs_data = 'model_output/decrease_CO_flux_case=0.3_bcs.csv'
    temps = np.arange(250, 370, 10)
    relh_lbs = [0.2, 0.6, 1.0]
    bcs = pd.read_csv(bcs_data, compression='gzip')
    bcs = bcs[bcs.alt==0.25]
    bcs['H2O_press'] = bcs['H2O'] * bcs['press']
    fig, axes = plt.subplots(1,1, figsize=(a4_x*0.6, a4_y*0.25))
    scatterplot(bcs, 'temp_lb', 'H2O', 'relh_lb', s=6, palette='viridis')
    lineplot(bcs, 'temp_lb', 'H2O', 'relh_lb', lw=0.9, palette='viridis')
    plt.yscale('log')
    plt.ylabel(r'Surface $H_2O$ mixing ratio', fontsize=8)
    plt.xlabel(r'Surface Temperature [K]', fontsize=8)
    plt.ylim([1e-4, 1])
    plt.xticks(temps)
    # add legend
    f = lambda c: plt.plot([],[], color=c, lw=2)[0]
    palette = sns.color_palette('viridis', 3)
    handles = []
    handles += [f(palette[i]) for i in range(3)]
    labels = [str(x) for x in relh_lbs]
    plt.legend(handles, labels, title='Relative Humidity', fontsize=8, 
        frameon=False, title_fontsize=8)
    if scenario == 'increase_O2':
        plt.savefig('increase_O2_H2O_mixrat_vs_temp.pdf', bbox_inches='tight')
    else:
        plt.savefig('decrease_CO_H2O_mixrat_vs_temp.pdf', bbox_inches='tight')
    plt.close('all')


@print_name
def mixrat_lb_vs_O2_flux_lb(scenario='increase_O2'):
    '''Plots model resulting mixrat at the LB vs the prescribed O2 flux
    using temperature as color hue for various species''' 
    if scenario == 'increase_O2':
        data_mixrat = 'model_output/increase_O2_flux_case=0.3_some_species_mixrat_lb.csv'
        data_cols = 'model_output/increase_O2_flux_case=0.3_some_species_columns.csv'
        flux = 'O2_flux_lb'
    else:
        data_mixrat = 'model_output/decrease_CO_flux_case=0.3_some_species_mixrat_lb.csv'
        data_cols = 'model_output/decrease_CO_flux_case=0.3_some_species_columns.csv'
        flux = 'CO_flux_lb'
    # read and filter data
    data_mixrat = pd.read_csv(data_mixrat, compression='gzip')
    data_cols = pd.read_csv(data_cols, compression='gzip')
    data_mixrat = filter_data(data_mixrat, scenario=scenario, rh='all')
    data_cols = filter_data(data_cols, scenario=scenario, rh='all')
    temps = np.arange(250, 370, 10.)
    rhs = data_mixrat.relh_lb.unique()
    for rh in rhs:
        data_mixrat_rh = data_mixrat[data_mixrat.relh_lb == rh]
        data_cols_rh = data_cols[data_cols.relh_lb == rh]
        fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
        species = ['O2', 'CH4', 'O3', 'H2', 'OH',  'CO']
        ylabels = [r'$O_2$', r'$CH_4$', r'$O_3$', r'$H_2$', r'$OH$', r'$CO$']
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            # plot O3 column
            if i == 2:
                scatterplot(data_cols_rh, flux, species[i], 'temp_lb', s=2)
                plt.ylabel(ylabels[i] + ' column ' + r'[$/cm^2$]',
                    fontsize=7)
            # plot species surface mixing ratio
            else:
                scatterplot(data_mixrat_rh, flux, species[i], 'temp_lb', s=2)
                plt.ylabel(ylabels[i] + ' surface mixing ratio', fontsize=7)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='x', which='major', direction='in')
            plt.tick_params(axis='x', which='minor', labelsize=4, direction='in')
            ax.xaxis.set_minor_formatter(format_minor_ticks)
            if i>3:
                plt.xlabel(flux_labels[scenario], fontsize=7)

        plt.subplots_adjust(hspace=0, wspace=0.25)
        add_colorbar(fig, axes, temps, 'Surface Temperature [K]', pad=0.1,
            ticks=temps)
        if scenario == 'increase_O2':
            fig.savefig(f'increase_O2_species_mixrat_lb_rh={rh}.pdf',
                bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_species_mixrat_lb_rh={rh}.pdf',
                bbox_inches='tight')
        plt.close('all')

@print_name
def mixrat_lb_vs_O2_flux_lb_rh(scenario='increase_O2'):
    '''Plots model resulting mixrat at the LB vs the prescribed O2 flux
    for various species using relative humidity as the color hue'''
    if scenario == 'increase_O2':
        data_mixrat = 'model_output/increase_O2_flux_case=0.3_some_species_mixrat_lb.csv'
        data_cols = 'model_output/increase_O2_flux_case=0.3_some_species_columns.csv'
        flux = 'O2_flux_lb'
    else:
        data_mixrat = 'model_output/decrease_CO_flux_case=0.3_some_species_mixrat_lb.csv'
        data_cols = 'model_output/decrease_CO_flux_case=0.3_some_species_columns.csv'
        flux = 'CO_flux_lb'
    # read and filter data
    data_mixrat = pd.read_csv(data_mixrat, compression='gzip')
    data_cols = pd.read_csv(data_cols, compression='gzip')
    data_mixrat = filter_data(data_mixrat, rh='all', scenario=scenario)
    data_cols = filter_data(data_cols, rh='all', scenario=scenario)
    temps = np.arange(250, 370, 10.)
    rhs = data_mixrat['relh_lb'].unique()
    species = ['O2', 'CH4', 'O3', 'H2', 'OH',  'CO']
    ylabels = [r'$O_2$', r'$CH_4$', r'$O_3$', r'$H_2$', r'$OH$', r'$CO$']

    # make a plot for each temperature
    for temp in temps:
        fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
        plt.suptitle(f'Surface Temperature = {temp}K', fontsize=8, y=0.92)
        axes_flat = axes.flatten()
        data_mixrat_temp = data_mixrat[data_mixrat.temp_lb == temp]
        data_cols_temp = data_cols[data_cols.temp_lb == temp]
        for i, ax in enumerate(axes_flat):
            plt.sca(ax)
            # plot O3 column
            if i == 2:
                scatterplot(data_cols_temp, flux ,species[i], 'relh_lb',
                    s=2, palette='viridis')
                plt.ylabel(ylabels[i] + ' column ' + r'[$/cm^2$]',
                     fontsize=7)
            # plot surface mixing ratio
            else:
                scatterplot(data_mixrat_temp, flux ,species[i], 'relh_lb',
                    s=2, palette='viridis')
                plt.ylabel(ylabels[i] + ' surface mixing ratio', fontsize=7)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='x', which='major', direction='in')
            plt.tick_params(axis='x', which='minor', labelsize=4, direction='in')
            ax.xaxis.set_minor_formatter(format_minor_ticks)
            if i>3:
                plt.xlabel(flux_labels[scenario], fontsize=7)

        plt.subplots_adjust(hspace=0, wspace=0.25)
        add_horizontal_legend(fig, axes, rhs, x=-0.2, y=0.1) 
        if scenario == 'increase_O2':
            fig.savefig(f'increase_O2_species_mixrat_lb_rh_temp={temp}.pdf',
                bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_species_mixrat_lb_rh_temp={temp}.pdf',
                bbox_inches='tight')
        plt.close('all')

@print_name
def mixrat_profiles(scenario='increase_O2'):
    '''Plots model resulting mixrat profiles for key species before, during and after
    the GOE using temperature as color hue'''
    if scenario == 'increase_O2':
        data_profiles = 'model_output/increase_O2_flux_case=0.3_some_species_profiles.csv'
        flux = 'O2_flux_lb'
    else:
        data_profiles = 'model_output/decrease_CO_flux_case=0.3_some_species_profiles.csv'
        flux = 'CO_flux_lb'

    # read and filter data
    data = pd.read_csv(data_profiles, compression='gzip')
    data = data[data.converged]
    temps = np.arange(250, 370, 10.)
    rhs = [0.2, 0.6, 1.0]

    species = ['O2', 'O3', 'CH4', 'OH', 'H2O', 'H2', 'CO']
    fluxes =  data[flux].unique()
    ylabels = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$OH$',  r'$H_{2}O$', r'$H_{2}$',
        r'$CO$']
    subtitles = ['Before GOE', 'During GOE', 'After GOE']

    for rh in rhs:
        fig, axes = plt.subplots(7,3, figsize=(a4_x*0.85, a4_y*0.9), sharex='row',
            sharey='col')
        data_rh = data[data.relh_lb == rh]
        for i, sp in enumerate(species):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_rh[data_rh[flux] == fluxes[j]]
                lineplot(data_flux, sp, 'alt', 'temp_lb', lw=0.8)
                plt.xscale('log')
                if j == 0:
                    plt.ylabel('Altitude [Km];\n Species:' + ylabels[i], fontsize=8)
                else:
                    axes[i][j].yaxis.set_ticklabels([])
                if i == 0:
                    axes[i][j].title.set_text(subtitles[j])
                    axes[i][j].title.set_size(8)
                if i == 6:
                    plt.xlabel('Mixing Ratio;\n' + 
                        flux_labels[scenario][:-12] + ':' +
                            sci_notation(fluxes[j]) + r'$/cm^2/s$', fontsize=7)

        plt.subplots_adjust(wspace=0, hspace=0.25)
        add_colorbar(fig, axes.flatten(), temps, 'Surface Temperature [K]',
            fraction=0.04, pad=0.07, aspect=70, ticks=temps)
        if scenario == 'increase_O2':
            fig.savefig(f'increase_O2_species_profiles_rh={rh}.pdf',
                bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_species_profiles_rh={rh}.pdf',
                bbox_inches='tight')
        plt.close('all')

@print_name
def mixrat_profiles_rh(scenario='increase_O2'):
    '''Plots model resulting mixrat profiles for key species before, during and after
    the GOE using relative humidity as color hue'''
    if scenario == 'increase_O2':
        data_profiles = 'model_output/increase_O2_flux_case=0.3_some_species_profiles.csv'
        flux = 'O2_flux_lb'
    else:
        data_profiles = 'model_output/decrease_CO_flux_case=0.3_some_species_profiles.csv'
        flux = 'CO_flux_lb'

    # read and filter data
    data = pd.read_csv(data_profiles, compression='gzip')
    data = data[data.converged]
    temps = np.arange(250, 370, 10.)
    rhs = [0.2, 0.6, 1.0]

    species = ['O2', 'O3', 'CH4', 'OH', 'H2O', 'H2', 'CO']
    fluxes =  data[flux].unique()
    ylabels = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$OH$',  r'$H_{2}O$', r'$H_{2}$',
        r'$CO$']
    subtitles = ['Before GOE', 'During GOE', 'After GOE']

    for temp in temps:
        fig, axes = plt.subplots(7,3, figsize=(a4_x*0.85, a4_y*0.8), sharex='row',
            sharey='col')
        data_temp = data[data.temp_lb == temp]
        for i, sp in enumerate(species):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_temp[data_temp[flux] == fluxes[j]]
                lineplot(data_flux, sp, 'alt', 'relh_lb', lw=0.8, palette='viridis')
                plt.xscale('log')
                if j == 0:
                    plt.ylabel('Altitude [Km];\n Species:' + ylabels[i], fontsize=8)
                else:
                    axes[i][j].yaxis.set_ticklabels([])
                if i == 0:
                    axes[i][j].title.set_text(subtitles[j])
                    axes[i][j].title.set_size(8)
                if i == 6:
                    plt.xlabel('Mixing Ratio;\n' + 
                        flux_labels[scenario][:-12] + ':' +
                            sci_notation(fluxes[j]) + r'$/cm^2/s$', fontsize=7)

        plt.subplots_adjust(wspace=0, hspace=0.25)
        add_horizontal_legend(fig, axes, rhs, y=0.08) 
        if scenario == 'increase_O2':
            fig.savefig(f'increase_O2_species_profiles_temp={temp}.pdf',
                bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_species_profiles_temp={temp}.pdf',
                bbox_inches='tight')
        plt.close('all')

@print_name
def rates_profiles(scenario='increase_O2'):
    '''Plots model resulting rate profiles for key reactions before, during and after
    the GOE using temperature as color hue'''
    if scenario == 'increase_O2':
        data_rates = 'model_output/increase_O2_flux_case=0.3_some_reaction_profiles.csv'
        flux = 'O2_flux_lb'
    else:
        data_rates = 'model_output/decrease_CO_flux_case=0.3_some_reaction_profiles.csv'
        flux = 'CO_flux_lb'
    # read and filter data
    data = pd.read_csv(data_rates, compression='gzip')
    data = data[data.converged]
    temps = np.arange(250, 370, 10.)
    rhs = [0.2, 0.6, 1.0]
    
    reactions = ['H2O + HV = H + OH', 'O2 + HV = O + O', 'O3 + HV = O2 + O', 
        'O + O = O2', 'O + O3 = O2 + O2', 'O + O2 = O3', 
        'CH4 + OH = CH3 + H2O']
    reaction_formulas = {'O2 + HV = O + O': r'$O_2 + hv \rightarrow O + O$',
        'O3 + HV = O2 + O': r'$O_3 + hv \rightarrow O_2 + O$', 
        'O + O = O2': r'$O + O \rightarrow O_2$',
        'O + O2 = O3': r'$O + O_2 \rightarrow O_3$',
        'O + O3 = O2 + O2': r'$O + O_3 \rightarrow O_2 + O_2$', 
        'H2O + HV = H + OH': r'$H_{2}O + hv \rightarrow H + OH$',
        'CH4 + OH = CH3 + H2O': r'$CH_4 + OH \rightarrow CH_3 + H_{2}O$'}
    fluxes =  data[flux].unique()
    subtitles = ['Before GOE', 'During GOE', 'After GOE']
    for rh in rhs:
        fig, axes = plt.subplots(7,3, figsize=(a4_x*0.85, a4_y*0.9), sharex='row',
            sharey='col')
        data_rh = data[data.relh_lb == rh]
        for i, rx in enumerate(reactions):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_rh[data_rh[flux] == fluxes[j]]
                lineplot(data_flux, rx, 'alt', 'temp_lb', lw=0.8)
                plt.xscale('log')
                if j == 0:
                    plt.ylabel('Altitude [Km];\n' + reaction_formulas[reactions[i]],
                        fontsize=7)
                else:
                    axes[i][j].yaxis.set_ticklabels([])
                if i == 0:
                    axes[i][j].title.set_text(subtitles[j])
                    axes[i][j].title.set_size(8)
                if i == 6:
                    plt.xlabel(r'Rate [$/cm^{3}/ s$]'+';\n' + 
                        flux_labels[scenario][:-12] + ':' +
                            sci_notation(fluxes[j]) + r'$/cm^2/s$', fontsize=7)

        plt.subplots_adjust(wspace=0, hspace=0.25)
        add_colorbar(fig, axes.flatten(), temps, 'Surface Temperature [K]',
            fraction=0.04, pad=0.07, aspect=70, ticks=temps)  
        if scenario == 'increase_O2':
            fig.savefig(f'increase_O2_reaction_profiles_rh={rh}.pdf', bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_reaction_profiles_rh={rh}.pdf', bbox_inches='tight')
        plt.close('all')

@print_name
def rates_profiles_rh(scenario='increase_O2'):
    '''Plots model resulting rate profiles for key reactions before, during and after
    the GOE using relative_humidity as color hue'''
    if scenario == 'increase_O2':
        data_rates = 'model_output/increase_O2_flux_case=0.3_some_reaction_profiles.csv'
        flux = 'O2_flux_lb'
    else:
        data_rates = 'model_output/decrease_CO_flux_case=0.3_some_reaction_profiles.csv'
        flux = 'CO_flux_lb'
    # read and filter data
    data = pd.read_csv(data_rates, compression='gzip')
    data = data[data.converged]
    temps = np.arange(250, 370, 10.)
    rhs = [0.2, 0.6, 1.0]
    
    reactions = ['H2O + HV = H + OH', 'O2 + HV = O + O', 'O3 + HV = O2 + O', 
        'O + O = O2', 'O + O3 = O2 + O2', 'O + O2 = O3', 
        'CH4 + OH = CH3 + H2O']
    reaction_formulas = {'O2 + HV = O + O': r'$O_2 + hv \rightarrow O + O$',
        'O3 + HV = O2 + O': r'$O_3 + hv \rightarrow O_2 + O$', 
        'O + O = O2': r'$O + O \rightarrow O_2$',
        'O + O2 = O3': r'$O + O_2 \rightarrow O_3$',
        'O + O3 = O2 + O2': r'$O + O_3 \rightarrow O_2 + O_2$', 
        'H2O + HV = H + OH': r'$H_{2}O + hv \rightarrow H + OH$',
        'CH4 + OH = CH3 + H2O': r'$CH_4 + OH \rightarrow CH_3 + H_{2}O$'}
    fluxes =  data[flux].unique()
    subtitles = ['Before GOE', 'During GOE', 'After GOE']
    for temp in temps:
        fig, axes = plt.subplots(7,3, figsize=(a4_x*0.8, a4_y*0.8), sharex='row',
            sharey='col')
        data_temp = data[data.temp_lb == temp]
        for i, rx in enumerate(reactions):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_temp[data_temp[flux] == fluxes[j]]
                lineplot(data_flux, rx, 'alt', 'relh_lb', lw=0.8, palette='viridis')
                plt.xscale('log')
                if j == 0:
                    plt.ylabel('Altitude [Km];\n' + reaction_formulas[reactions[i]],
                        fontsize=7)
                else:
                    axes[i][j].yaxis.set_ticklabels([])
                if i == 0:
                    axes[i][j].title.set_text(subtitles[j])
                    axes[i][j].title.set_size(8)
                if i == 6:
                    plt.xlabel(r'Rate [$/cm^{3}/ s$]'+';\n' + 
                        flux_labels[scenario][:-12] + ':' +
                            sci_notation(fluxes[j]) + r'$/cm^2/s$', fontsize=7)

        plt.subplots_adjust(wspace=0, hspace=0.25)
        add_horizontal_legend(fig, axes, rhs) 
        if scenario == 'increase_O2':
            fig.savefig(f'increase_O2_reaction_profiles_temp={temp}.pdf', bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_reaction_profiles_temp={temp}.pdf', bbox_inches='tight')
        plt.close('all')

@print_name
def radiative_flux(scenario='increase_O2'):
    '''Plots UV photon flux at different altitudes using temperature as color 
        hue'''
    if scenario == 'increase_O2':
        data_radiative_flux = 'model_output/increase_O2_flux_case=0.3_radiation.csv'
        flux = 'O2_flux_lb'
    else:
        data_radiative_flux = 'model_output/decrease_CO_flux_case=0.3_radiation.csv'
        flux = 'CO_flux_lb'
    # read and filter data
    temps = np.arange(250, 370, 10.)
    rad_flux = pd.read_csv(data_radiative_flux, compression='gzip')
    rad_flux = filter_data(rad_flux, rh='all', scenario=scenario)
    rad_flux = rad_flux[rad_flux.temp_lb.isin(temps)]

    fluxes = rad_flux[flux].unique()
    subtitles = ['Before GOE', 'During GOE', 'After GOE']
    rhs = [0.2, 0.6, 1.0]
    # altitudes to plot
    alts = ['60.25', '40.25', '20.25', '0.25']
    for rh in rhs:
        fig, axes = plt.subplots(4,3, figsize=(a4_x*0.85, a4_y*0.5), sharex='row',
            sharey='col')
        data_rh = rad_flux[rad_flux.relh_lb == rh]
        for i, alt in enumerate(alts):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_rh[data_rh[flux] == fluxes[j]]
                lineplot(data_flux, 'wavl', alt, 'temp_lb', lw=0.8)
                data_flux_290 = data_flux[data_flux.temp_lb == 290]
                plt.plot(data_flux_290.wavl, data_flux_290['99.75'], color='k',
                    lw=0.8, label='TOA')
                plt.yscale('log')
                plt.ylim([1e-3, 1e16])
                if j == 0:
                    plt.ylabel('Flux\n' + r'$hv \, cm^{-2} s^{-1} nm^{-1}$', fontsize=7)
                    plt.text(0.45, 0.25, f'Altitude:{alt} km', fontsize=7, 
                        transform=axes[i][j].transAxes)
                    if i == 0:
                        plt.legend(fontsize=7, frameon=False)

                axes[0][1].yaxis.set_ticklabels([])
                axes[0][2].yaxis.set_ticklabels([])
                if i == 0:
                    axes[i][j].title.set_text(subtitles[j])
                    axes[i][j].title.set_size(8)
                axes[i][j].tick_params(axis="x",direction="in")
                if i == 3:
                    plt.xlabel('Wavelength [nm]\n' +
                        flux_labels[scenario][:-12] + ':' +
                            sci_notation(fluxes[j]) + r'$/cm^2/s$', fontsize=7)
        plt.subplots_adjust(wspace=0, hspace=0)
        add_colorbar(fig, axes.flatten(), temps, 'Surface Temperature [K]',
            fraction=0.04, pad=0.1, aspect=70, ticks=temps)
        if scenario == 'increase_O2': 
            fig.savefig(f'increase_O2_radiative_flux_rh={rh}.pdf',
                bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_radiative_flux_rh={rh}.pdf',
                bbox_inches='tight')
        plt.close('all')

@print_name
def radiative_flux_rh(scenario='increase_O2'):
    '''Plots UV photon flux at different altitudes using relative humidity 
        as color  hue'''
    if scenario == 'increase_O2':
        data_radiative_flux = 'model_output/increase_O2_flux_case=0.3_radiation.csv'
        flux = 'O2_flux_lb'
    else:
        data_radiative_flux = 'model_output/decrease_CO_flux_case=0.3_radiation.csv'
        flux = 'CO_flux_lb'
    # read and filter data
    temps = np.arange(250, 370, 10.)
    rad_flux = pd.read_csv(data_radiative_flux, compression='gzip')
    rad_flux = filter_data(rad_flux, rh='all', scenario=scenario)
    rad_flux = rad_flux[rad_flux.temp_lb.isin(temps)]

    fluxes = rad_flux[flux].unique()
    subtitles = ['Before GOE', 'During GOE', 'After GOE']
    rhs = [0.2, 0.6, 1.0]
    # altitudes to plot
    alts = ['60.25', '40.25', '20.25', '0.25']
    for temp in temps:
        fig, axes = plt.subplots(4,3, figsize=(a4_x*0.85, a4_y*0.5), sharex='row',
            sharey='col')
        data_temp = rad_flux[rad_flux.temp_lb == temp]
        for i, alt in enumerate(alts):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_temp[data_temp[flux] == fluxes[j]]
                lineplot(data_flux, 'wavl', alt, 'relh_lb', lw=0.8, palette='viridis')
                data_flux_06 = data_flux[data_flux.relh_lb == 0.6]
                plt.plot(data_flux_06.wavl, data_flux_06['99.75'], color='k',
                    lw=0.8, label='TOA')
                plt.yscale('log')
                plt.ylim([1e-3, 1e16])
                if j == 0:
                    plt.ylabel('Flux\n' + r'$hv \, cm^{-2} s^{-1} nm^{-1}$', fontsize=7)
                    plt.text(0.45, 0.25, f'Altitude:{alt} km', fontsize=7, 
                        transform=axes[i][j].transAxes)
                    if i == 0:
                        plt.legend(fontsize=7, frameon=False)

                axes[0][1].yaxis.set_ticklabels([])
                axes[0][2].yaxis.set_ticklabels([])
                if i == 0:
                    axes[i][j].title.set_text(subtitles[j])
                    axes[i][j].title.set_size(8)
                axes[i][j].tick_params(axis="x",direction="in")
                if i == 3:
                    plt.xlabel('Wavelength [nm]\n' +
                        flux_labels[scenario][:-12] + ':' +
                            sci_notation(fluxes[j]) + r'$/cm^2/s$', fontsize=7)
        plt.subplots_adjust(wspace=0, hspace=0)
        add_horizontal_legend(fig, axes, rhs, x=0.2, y=0.12)
        if scenario == 'increase_O2': 
            fig.savefig(f'increase_O2_radiative_flux_temp={temp}.pdf',
                bbox_inches='tight')
        else:
            fig.savefig(f'decrease_CO_radiative_flux_temp={temp}.pdf',
                bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    boundary_conditions()
    gregory_case1_vs_this_study()
    O3_and_H2O_after_GOE()
    OH_main_sources()
    OH_main_sources_O3_and_H2O_after_GOE_colorbars()
    H2O_mixrat_vs_temp(scenario='increase_O2')
    H2O_mixrat_vs_temp(scenario='decrease_CO')
    mixrat_lb_vs_O2_flux_lb(scenario='increase_O2')
    mixrat_lb_vs_O2_flux_lb(scenario='decrease_CO')
    mixrat_lb_vs_O2_flux_lb_rh(scenario='increase_O2')
    mixrat_lb_vs_O2_flux_lb_rh(scenario='decrease_CO')
    mixrat_profiles(scenario='increase_O2')
    mixrat_profiles(scenario='decrease_CO')
    mixrat_profiles_rh(scenario='increase_O2')
    mixrat_profiles_rh(scenario='decrease_CO')
    rates_profiles(scenario='increase_O2')
    rates_profiles(scenario='decrease_CO')
    rates_profiles_rh(scenario='increase_O2')
    rates_profiles_rh(scenario='decrease_CO')
    radiative_flux(scenario='increase_O2')
    radiative_flux(scenario='decrease_CO')
    radiative_flux_rh(scenario='increase_O2')
    radiative_flux_rh(scenario='decrease_CO')