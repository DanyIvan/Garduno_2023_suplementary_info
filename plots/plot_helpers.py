import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import math as m


def add_colorbar(fig, axes, values, label, palette='coolwarm', fraction=0.05,
    pad=0.08, aspect=70, fontsize=8, ticks=[]):
    '''Adds a colorbar to a figure'''
    cmap =  sns.color_palette(palette, len(values), as_cmap=True)
    norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=axes, orientation='horizontal', fraction=fraction, pad=pad,
        aspect=aspect)
    cb.set_label(label, fontsize=fontsize)
    if len(ticks) != 0:
        cb.set_ticks(ticks)

def add_horizontal_legend(fig, axes, values, palette='viridis',
    title='Relative Humidity:', x=0.5, y=0.08): 
    # create handles for legend  
    f = lambda c: plt.plot([],[], color=c, lw=2)[0]
    palette = sns.color_palette(palette, len(values))
    handles = []
    handles += [f(palette[i]) for i in range(len(values))]
    labels = [str(x) for x in values]
    if len(axes.shape) == 1:
        axbox = axes[-1].get_position()
    else:
        axbox = axes[-1][1].get_position()
    # create legend
    leg = fig.legend(handles, labels, loc='lower center', ncol=len(values),
        title=title, fontsize=8,
        bbox_to_anchor=[axbox.x0 + x*axbox.width, axbox.y0-y], 
        bbox_transform=fig.transFigure, frameon=False, title_fontsize=8)
    c = leg.get_children()[0]
    title = c.get_children()[0]
    hpack = c.get_children()[1]
    c._children = [hpack]
    hpack._children = [title] + hpack.get_children()

def scatterplot(data, x, y, hue, s=10, palette='coolwarm', alpha=1,
    labels=[]):
    '''Makes a scatterplot with a hue'''
    values = data[hue].unique()
    if type(palette) != list:
        palette = sns.color_palette(palette, len(values))
    groups = data.groupby(hue)
    for i, key in enumerate(groups.groups.keys()):
        group = groups.get_group(key)
        label = labels[i] if len(labels) > 0 else None
        plt.scatter(group[x], group[y], color=palette[i], s=s, alpha=alpha,
            label=label)

def lineplot(data, x, y, hue, palette='coolwarm', lw=1, alpha=1, labels=[]):
    '''Makes a lineplot with a hue'''
    values = data[hue].unique()
    if type(palette) != list:
        palette = sns.color_palette(palette, len(values))
    groups = data.groupby(hue)
    for i, key in enumerate(groups.groups.keys()):
        group = groups.get_group(key)
        label = labels[i] if len(labels) > 0 else None
        plt.plot(group[x], group[y], color=palette[i], lw=lw, label=label,
            alpha=alpha)

def filter_data(data, rh=0.6, scenario='increase_O2'):
    '''Filters surface O2 and CO flux range, considering only converged solutions'''
    data = data[data.converged]
    if scenario == 'increase_O2':
        data = data[data.O2_flux_lb > 1e11]
        data = data[data.O2_flux_lb < 1.1e12]
    else:
        data = data[data.CO_flux_lb > 0.5e11]
        data = data[data.CO_flux_lb < 6e11]
        pass
    # if rh == 'all' keep all relative humidities
    if rh != 'all':
        data = data[data.relh_lb == rh]
    return data

def format_minor_ticks(x, pos=None):
    '''Formater for minor ticks in log plots'''
    return str(x)[0]

def sci_notation(num, decimal_digits=2):
    '''Converts a number to a string in scientific notation'''
    exponent = int(m.floor(m.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    precision = decimal_digits
    return r"${}\times 10^{{{}}}$".format(coeff, exponent)

def print_name(func):
    '''Decorator to print name of runing function'''
    def inner(*args, **kwargs):
        print('Running ', func.__name__)
        result = func(*args, **kwargs)
        return result
    return inner

def subplots_centered(nrows, ncols, figsize, nfigs):
    """
    Source: https://stackoverflow.com/questions/53361373/center-the-third-
        subplot-in-the-middle-of-second-row-python
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, axs

