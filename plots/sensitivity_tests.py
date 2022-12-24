import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from plot_helpers import add_colorbar, scatterplot, lineplot,  print_name,\
    filter_data, format_minor_ticks, sci_notation, format_minor_ticks

folder = 'sensitivity_tests/'

# size of figure to fit in a4 page size
a4_x, a4_y = (8.27, 11.69)

# set ticks size
mpl.rc('xtick', labelsize=6) 
mpl.rc('ytick', labelsize=6)

flux_labels = {'increase_O2': r'Surface $O_2$ flux [$/cm^2/s$]',
    'decrease_CO': r'Surface $CO$ flux [$/cm^2/s$]'}

@print_name
def colder_stratosphere_test():
    '''Make plots to explore our results with a colder stratosphere''' 
    scenario = 'increase_O2'
    data_mixrat = 'model_output/increase_O2_flux_case=0.3_some_species_mixrat_lb.csv'
    data_cols = 'model_output/increase_O2_flux_case=0.3_some_species_columns.csv'
    data_profiles = 'model_output/increase_O2_flux_case=0.3_some_species_profiles.csv'
    flux = 'O2_flux_lb'

    data_mixrat_cold_strat = 'model_output/increase_O2_flux_case=0.3_colder' +\
        '_stratosphere_some_species_mixrat_lb.csv'
    data_cols_cold_strat = 'model_output/increase_O2_flux_case=0.3_colder_' +\
        'stratosphere_some_species_columns.csv'
    data_profiles_cold_strat = 'model_output/increase_O2_flux_case=0.3_' + \
        'colder_stratosphere_some_species_profiles.csv'


    # read and filter data
    data_mixrat_cold_strat = pd.read_csv(data_mixrat_cold_strat, 
        compression='gzip')
    data_cols_cold_strat = pd.read_csv(data_cols_cold_strat,
        compression='gzip')
    data_mixrat_cold_strat = filter_data(data_mixrat_cold_strat, 
        scenario=scenario, rh='all')
    data_cols_cold_strat = filter_data(data_cols_cold_strat,
        scenario=scenario, rh='all')
    data_profiles_cold_strat = pd.read_csv(data_profiles_cold_strat, 
        compression='gzip')
    data_profiles_cold_strat = data_profiles_cold_strat[
        data_profiles_cold_strat.converged]
    data_profiles_cold_strat = data_profiles_cold_strat[
        data_profiles_cold_strat.relh_lb == 0.6]

    data_mixrat = pd.read_csv(data_mixrat, compression='gzip')
    data_cols = pd.read_csv(data_cols, compression='gzip')
    data_mixrat = filter_data(data_mixrat, scenario=scenario, rh=0.6)
    data_cols = filter_data(data_cols, scenario=scenario, rh=0.6)
    data_profiles = pd.read_csv(data_profiles, compression='gzip')
    data_profiles = data_profiles[data_profiles.converged]
    data_profiles = data_profiles[data_profiles.relh_lb == 0.6]

    temps = np.arange(250, 370, 10.)
    fluxes =  data_profiles[flux].unique()
    subtitles = ['Before GOE', 'During GOE', 'After GOE']
    species = ['O2', 'CH4', 'O3', 'H2', 'OH',  'CO']
    ylabels = [r'$O_2$', r'$CH_4$', r'$O_3$', r'$H_2$', r'$OH$', r'$CO$']

    # plot mixrat lb with temp as color
    fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        plt.sca(ax)
        # plot O3 column
        if i == 2:
            scatterplot(data_cols_cold_strat, flux, species[i], 'temp_lb', s=2)
            plt.ylabel(ylabels[i] + ' column ' + r'[$/cm^2$]',
                fontsize=7)
        # plot species surface mixing ratio
        else:
            scatterplot(data_mixrat_cold_strat, flux, species[i], 'temp_lb', s=2)
            plt.ylabel(ylabels[i] + ' surface mixing ratio', fontsize=7)
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
    fig.savefig(folder + f'increase_O2_species_mixrat_lb_colder_stratosphere.pdf',
        bbox_inches='tight')

    # make plots to compare to 215k stratosphere
    for temp in temps:
        data_mixrat_temp = data_mixrat[data_mixrat.temp_lb == temp]
        data_cols_temp = data_cols[data_cols.temp_lb == temp]
        data_profiles_temp = data_profiles[data_profiles.temp_lb == temp]
        data_mixrat_cold_strat_temp = data_mixrat_cold_strat[
                data_mixrat_cold_strat.temp_lb == temp]
        data_cols_cold_strat_temp = data_cols_cold_strat[
                data_cols_cold_strat.temp_lb == temp]
        data_profiles_cold_strat_temp = data_profiles_cold_strat[
            data_profiles_cold_strat.temp_lb == temp]
        
        # plot surface mixing ratios
        fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            plt.scatter(data_mixrat_temp.O2_flux_lb, data_mixrat_temp[species[i]],
                s=2, color='k', label='215k')
            plt.scatter(data_mixrat_cold_strat_temp.O2_flux_lb, 
                data_mixrat_cold_strat_temp[species[i]],
                s=2, color='b', label='175k')
            plt.ylabel(ylabels[i] + ' surface mixing ratio', fontsize=7)
            if i == 0:
                plt.legend(title='Stratosphere temperature', fontsize=6,
                    title_fontsize=6)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='x', which='major', direction='in')
            plt.tick_params(axis='x', which='minor', labelsize=4, direction='in')
            ax.xaxis.set_minor_formatter(format_minor_ticks)
            if i>3:
                plt.xlabel(flux_labels[scenario], fontsize=7)

        plt.subplots_adjust(hspace=0, wspace=0.25)
        fig.savefig(folder + 'increase_O2_species_mixrat_lb_colder_stratosphere_test_' +\
            f'temp={temp}.pdf', bbox_inches='tight')
        
         # plot columns
        fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            plt.scatter(data_cols_temp.O2_flux_lb, data_cols_temp[species[i]],
                s=2, color='k', label='215k')
            plt.scatter(data_cols_cold_strat_temp.O2_flux_lb, 
                data_cols_cold_strat_temp[species[i]],
                s=2, color='b', label='175k')
            plt.ylabel(ylabels[i] + ' column ' + r'[$/cm^2$]',
                fontsize=7)
            if i == 0:
                plt.legend(title='Stratosphere temperature', fontsize=6,
                    title_fontsize=6)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='x', which='major', direction='in')
            plt.tick_params(axis='x', which='minor', labelsize=4, direction='in')
            ax.xaxis.set_minor_formatter(format_minor_ticks)
            if i>3:
                plt.xlabel(flux_labels[scenario], fontsize=7)

        plt.subplots_adjust(hspace=0, wspace=0.25)
        fig.savefig(folder + 'increase_O2_species_cols_colder_stratosphere_test' +\
            f'_temp={temp}.pdf', bbox_inches='tight')
        
        # plot profiles
        species_profiles = ['O2', 'O3', 'CH4', 'OH', 'H2O', 'H2', 'CO']
        ylabels_profiles = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$OH$',  
            r'$H_{2}O$', r'$H_{2}$', r'$CO$']
        fig, axes = plt.subplots(7,3, figsize=(a4_x*0.85, a4_y*0.9), sharex='row',
            sharey='col')   
        for i, sp in enumerate(species_profiles):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_profiles_temp[data_profiles_temp[flux] == fluxes[j]]
                data_flux_cold_strat = data_profiles_cold_strat_temp[
                    data_profiles_cold_strat_temp[flux] == fluxes[j]]
                plt.plot(data_flux[sp], data_flux.alt, color='k', lw=0.8, 
                    label='215k')
                plt.plot(data_flux_cold_strat[sp], data_flux_cold_strat.alt,
                    color='b', lw=0.8, label='0')
                plt.xscale('log')
                if j == 0:
                    plt.ylabel('Altitude [Km];\n Species:' + ylabels_profiles[i], fontsize=8)
                    if i==0:
                        plt.legend(title='Stratosphere temperature', fontsize=6,
                            title_fontsize=6)
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
        fig.savefig(folder + f'increase_O2_species_profiles_colder_stratosphere_test_{temp}.pdf',
            bbox_inches='tight')
        plt.close('all')

@print_name
def eddy_diff_sensitivity_test():
    '''Makes plots to explore our results with lower and higher eddy diffusivities''' 
    scenario = 'increase_O2'
    data_mixrat = 'model_output/increase_O2_flux_case=0.3_some_species_mixrat_lb.csv'
    data_cols = 'model_output/increase_O2_flux_case=0.3_some_species_columns.csv'
    data_profiles = 'model_output/increase_O2_flux_case=0.3_some_species_profiles.csv'
    flux = 'O2_flux_lb'

    data_mixrat_eddy_sens = 'model_output/increase_O2_flux_' +\
        'case=0.3_eddy_sensitivity_some_species_mixrat_lb.csv'
    data_cols_eddy_sens = 'model_output/increase_O2_flux_' +\
        'case=0.3_eddy_sensitivity_some_species_columns.csv'
    data_profiles_eddy_sens = 'model_output/increase_O2_flux_case=0.3_eddy_' +\
        'sensitivity_some_species_profiles.csv'


    # read and filter data
    data_mixrat_eddy_sens = pd.read_csv(data_mixrat_eddy_sens, 
        compression='gzip')
    data_cols_eddy_sens = pd.read_csv(data_cols_eddy_sens,
        compression='gzip')
    data_mixrat_eddy_sens = filter_data(data_mixrat_eddy_sens, 
        scenario=scenario, rh='all')
    data_cols_eddy_sens = filter_data(data_cols_eddy_sens,
        scenario=scenario, rh='all')
    data_profiles_eddy_sens = pd.read_csv(data_profiles_eddy_sens,
        compression='gzip')
    data_profiles_eddy_sens = data_profiles_eddy_sens[
        data_profiles_eddy_sens.converged]
    data_profiles_eddy_sens = data_profiles_eddy_sens[
        data_profiles_eddy_sens.relh_lb == 0.6]

    data_mixrat = pd.read_csv(data_mixrat, compression='gzip')
    data_cols = pd.read_csv(data_cols, compression='gzip')
    data_mixrat = filter_data(data_mixrat, scenario=scenario, rh=0.6)
    data_cols = filter_data(data_cols, scenario=scenario, rh=0.6)
    data_profiles = pd.read_csv(data_profiles, compression='gzip')
    data_profiles = data_profiles[data_profiles.converged]
    data_profiles = data_profiles[data_profiles.relh_lb == 0.6]

    temps = np.arange(250, 370, 10.)
    fluxes =  data_profiles[flux].unique()
    subtitles = ['Before GOE', 'During GOE', 'After GOE']
    species = ['O2', 'CH4', 'O3', 'H2', 'OH',  'CO']
    ylabels = [r'$O_2$', r'$CH_4$', r'$O_3$', r'$H_2$', r'$OH$', r'$CO$']

    # plot mixrat lb with temp as color
    eddy_increase_list = data_mixrat_eddy_sens.eddy_increase.unique()
    for eddy_increase in eddy_increase_list:
        data_mixrat_eddy = data_mixrat_eddy_sens[
            data_mixrat_eddy_sens.eddy_increase == eddy_increase]
        data_cols_eddy = data_cols_eddy_sens[
            data_cols_eddy_sens.eddy_increase == eddy_increase]
        fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            # plot O3 column
            if i == 2:
                scatterplot(data_cols_eddy, flux, species[i], 'temp_lb', s=2)
                plt.ylabel(ylabels[i] + ' column ' + r'[$/cm^2$]',
                    fontsize=7)
            # plot species surface mixing ratio
            else:
                scatterplot(data_mixrat_eddy, flux, species[i], 'temp_lb', s=2)
                plt.ylabel(ylabels[i] + ' surface mixing ratio', fontsize=7)
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
        fig.savefig(folder + f'increase_O2_species_mixrat_lb_eddy={eddy_increase}.pdf',
                bbox_inches='tight')
        plt.close('all')

    # compare results with eddy diff as color
    palette = ['#8db0fe', '#cd59df', '#f4987a']
    for temp in temps:
        data_mixrat_temp = data_mixrat[data_mixrat.temp_lb == temp]
        data_cols_temp = data_cols[data_cols.temp_lb == temp]
        data_mixrat_eddy_sens_temp = data_mixrat_eddy_sens[
                data_mixrat_eddy_sens.temp_lb == temp]
        data_cols_eddy_sens_temp = data_cols_eddy_sens[
                data_cols_eddy_sens.temp_lb == temp]
        data_profiles_temp = data_profiles[data_profiles.temp_lb == temp]
        data_profiles_eddy_sens_temp = data_profiles_eddy_sens[
            data_profiles_eddy_sens.temp_lb == temp]

        # plot mixing ratio at surface
        fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            plt.scatter(data_mixrat_temp.O2_flux_lb, data_mixrat_temp[species[i]],
                s=2, color='k', label='0')
            scatterplot(data_mixrat_eddy_sens_temp, flux, species[i], 
                'eddy_increase', labels=['-1', '1', '2'], palette=palette)
            plt.ylabel(ylabels[i] + ' surface mixing ratio', fontsize=7)
            if i == 0:
                plt.legend(title='Change in order\nof magnitude', fontsize=6,
                    title_fontsize=6)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='x', which='major', direction='in')
            plt.tick_params(axis='x', which='minor', labelsize=4, direction='in')
            ax.xaxis.set_minor_formatter(format_minor_ticks)
            if i>3:
                plt.xlabel(flux_labels[scenario], fontsize=7)

        plt.subplots_adjust(hspace=0, wspace=0.25)
        fig.savefig(folder + 'increase_O2_species_mixrat_lb_eddy_sensitivity' +\
            f'_temp={temp}.pdf', bbox_inches='tight')

        # plot columns
        fig, axes = plt.subplots(3,2, figsize=(a4_x*0.85, a4_y*0.5), sharex='col')
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            plt.sca(ax)
            plt.scatter(data_cols_temp.O2_flux_lb, data_cols_temp[species[i]],
                    s=2, color='k', label='0')
            scatterplot(data_cols_eddy_sens_temp, flux, species[i], 
                'eddy_increase', labels=['-1', '1', '2'], palette=palette)
            plt.ylabel(ylabels[i] + ' column ' + r'[$/cm^2$]',
                fontsize=7)
            if i == 0:
                plt.legend(title='Change in order\nof magnitude', fontsize=6,
                    title_fontsize=6)
            plt.yscale('log')
            plt.xscale('log')
            plt.tick_params(axis='x', which='major', direction='in')
            plt.tick_params(axis='x', which='minor', labelsize=4, direction='in')
            ax.xaxis.set_minor_formatter(format_minor_ticks)
            if i>3:
                plt.xlabel(flux_labels[scenario], fontsize=7)

        plt.subplots_adjust(hspace=0, wspace=0.25)
        fig.savefig(folder + 'increase_O2_species_cols_eddy_sensitivity' +\
            f'_temp={temp}.pdf', bbox_inches='tight')
        
        # plot profiles
        species_profiles = ['O2', 'O3', 'CH4', 'OH', 'H2O', 'H2', 'CO']
        ylabels_profiles = [r'$O_2$', r'$O_3$', r'$CH_4$', r'$OH$', r'$H_{2}O$',
            r'$H_{2}$', r'$CO$']
        fig, axes = plt.subplots(7,3, figsize=(a4_x*0.85, a4_y*0.9), sharex='row',
            sharey='col')   
        for i, sp in enumerate(species_profiles):
            for j in range(3):
                plt.sca(axes[i][j])
                data_flux = data_profiles_temp[data_profiles_temp[flux] == fluxes[j]]
                data_flux_eddy_sens = data_profiles_eddy_sens_temp[
                    data_profiles_eddy_sens_temp[flux] == fluxes[j]]
                plt.plot(data_flux[sp], data_flux.alt, color='k', lw=0.8, 
                    label='0')
                lineplot(data_flux_eddy_sens, sp, 'alt', 'eddy_increase', lw=0.8,
                    labels=['-1', '1', '2'], palette=palette)
                plt.xscale('log')
                if j == 0:
                    plt.ylabel('Altitude [Km];\n Species:' + ylabels_profiles[i], fontsize=8)
                    if i==0:
                        plt.legend(title='Change in order\nof magnitude', fontsize=6,
                            title_fontsize=6)
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
        fig.savefig(folder + f'increase_O2_species_profiles_eddy_sensitivity_temp={temp}.pdf',
            bbox_inches='tight')
        plt.close('all')


@print_name
def colorbars_eddy():
    '''
    Plots colorbars
    ''' 
    temps = np.arange(250, 370, 10)
    fig, axes = plt.subplots(1,1, figsize=(a4_x*0.5, a4_y*0.25))
    palette = ['#8db0fe', '#000000', '#cd59df', '#f4987a']
    cmaps = [sns.color_palette(f"light:{i}", len(temps), as_cmap=True)
        for i in palette]
    
    norm = mpl.colors.Normalize(vmin=min(temps), vmax=max(temps))
    for i, cmap in enumerate(cmaps[::-1]):
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
            ax=axes, orientation='horizontal', aspect=40)
        if i == 0:
            cb.set_ticks(temps)
            cb.set_label(label='Surface Temperature [K]', size=8)
        else:
            cb.set_ticks([])
        
    plt.savefig(folder + 'colorbars_eddy.pdf', bbox_inches='tight', transparent=True)

@print_name
def plot_eddy_diff_sensitivity_profiles():
    '''Plots eddy diffusivity profiles used in our  eddy difusivity sensitivity 
    test''' 

    bcs_data_original = 'model_output/increase_O2_flux_case=0.3_bcs.csv'
    data_original = pd.read_csv(bcs_data_original, compression='gzip')
    data_original = data_original[data_original.relh_lb == 0.6]

    bcs_data = 'model_output/increase_O2_flux_case=0.3_eddy_sensitivity_bcs.csv'
    data = pd.read_csv(bcs_data, compression='gzip')

    eddy_increase_list = data.eddy_increase.unique()

    fig, axes = plt.subplots(1,1,figsize=(a4_x * 0.5, a4_y * 0.3))
    palette = ['#8db0fe', '#cd59df', '#f4987a']
    temps = data_original.temp_lb.unique()
    labels = ['-1', '1', '2']
    last_color = lambda palette:sns.color_palette(palette, len(temps))[-1]
    lineplot(data_original, 'edd', 'alt', 'temp_lb', lw=0.7,
        palette='light:#000000') 
    data_original_360 = data_original[data_original.temp_lb == 360]
    plt.plot(data_original_360.edd, data_original_360.alt, label='0',
            color='k', lw=0.9, alpha=0.8) 
    for i, eddy_increase in enumerate(eddy_increase_list):
        data_eddy_increase = data[data.eddy_increase == eddy_increase]
        data_360 = data_eddy_increase[data_eddy_increase.temp_lb == 360]
        lineplot(data_eddy_increase, 'edd', 'alt', 'temp_lb', lw=0.7,
        palette='light:' + palette[i])
        plt.plot(data_360.edd, data_360.alt, label=labels[i],
            color=last_color("light:" + palette[i]), lw=0.9, alpha=0.8) 
    plt.xlabel(r'Eddy diffusivity [$cm^{2} s^{-1}$]',fontsize=8)
    plt.ylabel('Altitude [km]')
    plt.legend(title='Change in order\nof magnitude', fontsize=6,
                    title_fontsize=6)
    plt.xscale('log')

    fig.savefig(folder + 'eddy_sensitivity_test_profiles.pdf', bbox_inches='tight')
    plt.close('all')

@print_name
def plot_colder_stratosphere_temp_profiles():
    '''Plots temperature profiles used in our colder stratosphere test ''' 

    bcs_data = 'model_output/increase_O2_flux_case=0.3_colder_stratosphere_bcs.csv'
    data = pd.read_csv(bcs_data, compression='gzip')
    temps = np.arange(250, 370, 10.)
    data = data[data.temp_lb.isin(temps)]

    fig, axes = plt.subplots(1,1,figsize=(a4_x * 0.5, a4_y * 0.3))
    lineplot(data, 'T', 'alt', 'temp_lb', lw=0.7)
    plt.ylabel('Altitude [Km]', fontsize=8)
    plt.xlabel('Temperature [K]', fontsize=8)

    add_colorbar(fig, axes, temps, 'Surface Temperature [K]', pad=0.15,
        ticks=temps)
    fig.savefig(folder + 'colder_stratosphere_temp_profiles.pdf', bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    colder_stratosphere_test()
    eddy_diff_sensitivity_test()
    colorbars_eddy()
    plot_eddy_diff_sensitivity_profiles()
    plot_colder_stratosphere_temp_profiles()
