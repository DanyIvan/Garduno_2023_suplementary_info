import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


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
    palette = sns.color_palette(palette, len(values))
    groups = data.groupby(hue)
    for i, key in enumerate(groups.groups.keys()):
        group = groups.get_group(key)
        label = labels[i] if len(labels) > 0 else None
        plt.plot(group[x], group[y], color=palette[i], lw=lw, label=label,
            alpha=alpha)

