# flake8: noqa

from cycler import cycler
# box style
paper_style = {
    # Colour cycle
    'axes.prop_cycle': cycler(color=['#377eb8', '#ff7f00', '#984ea3',
                                     '#4daf4a', '#a65628', '#f781bf',
                                     '#999999', '#e41a1c', '#dede00']),

    # Line styles
    'lines.linewidth': 1.3,
    'lines.antialiased': True,

    # Error bars
    'errorbar.capsize': 3,  # length of end cap on error bars in pixels

    # Font
    'font.size': 18.0,

    # Axes
    'axes.linewidth': 1.5,
    'axes.titlesize': 'x-large',
    'axes.labelsize': 'large',
    'axes.spines.top': True,
    'axes.spines.right': True,

    # Ticks
    'xtick.major.size': 6,
    'xtick.minor.size': 4,
    'xtick.major.width': 1.5,
    'xtick.minor.width': 1.5,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 6,
    'xtick.labelsize': 'medium',
    'xtick.direction': 'in',
    'xtick.top': False,

    'ytick.major.size': 6,
    'ytick.minor.size': 4,
    'ytick.major.width': 1.5,
    'ytick.minor.width': 1.5,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 6,
    'ytick.labelsize': 'medium',
    'ytick.direction': 'in',
    'ytick.right': False,

    # Legend
    'legend.fancybox': True,
    'legend.fontsize': 'large',
    'legend.scatterpoints': 5,
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [8, 5.2],
    'figure.titlesize': 'large',

    # Images
    'image.cmap': 'magma',
    'image.origin': 'lower',

    # Saving
    'savefig.bbox': 'tight',
    'savefig.format': 'png',
}
