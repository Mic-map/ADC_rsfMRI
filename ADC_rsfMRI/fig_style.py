"""
Import this file to set output settings for report style

(c) https://github.com/Guigeek64/, June 2024

"""

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('default')  # reset
# you can list with: plt.style.available
plt.style.use(['seaborn-v0_8-white']) # you can combine sevaral
# This is the linewidth in latex. Shown in the compilation log
# by: \showthe\textwidth inside \begin{document}
latex_linewidth = 483.69687

# Custom style: adjust here the fonts and sizes
latex_params = {
                # Use LaTeX to write all text
                # "text.usetex": True,
                "font.family": "serif",
                # Use 10pt font in plots (a bit smaller than 11pt font in document)
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.labelpad": 2.0,
                # Make the legend/label fonts and pads a little smaller
                "legend.fontsize": 8,
                "legend.framealpha": 0.5,
                "legend.labelspacing": 0.2,
                "legend.borderpad": 0.2,
                "legend.borderaxespad": 0.3,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "xtick.major.pad": 2,
                "ytick.major.pad": 2,
                "axes.grid": False,
                # Plots
                "lines.linewidth": 1,
                "lines.markeredgewidth": 0.5,
                "lines.markersize": 1,
                "image.cmap" : 'cool',
                # Save with small borders
                "savefig.bbox": 'tight',
                "savefig.pad_inches": 0.03,
                "figure.autolayout" : True
                }
mpl.rcParams.update(latex_params)

#(5**.5 - 1)/2,
def fig_size(fraction=1, height_ratio=1, subplot=[1, 1]):
    """ Computes figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    fraction: float
            Fraction of the width which you wish the figure to occupy
    height_ratio: float
            Ratio between hight and width (golden by default)
    subplot: int list
            Sublots count
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt  = latex_linewidth * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in  = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * height_ratio * (subplot[0] / subplot[1])

    return (fig_width_in, fig_height_in)


"""
How to use:

from fig_style import fig_size
import matplotlib.pyplot as plt

# 0.49 to match \includegraphics[width=0.49\linewidth] in latex
fig, ax1 = plt.subplots(1, 1, figsize=fig_size(fraction=0.49))
ax1.plot(V, I, lw=2, c='b', label='Output')
ax1.legend(loc='lower left')
ax1.set_xlabel('Voltage $V$ [V]')
ax1.set_ylabel('Current $I$ [A]')
fig.tight_layout() # Warning: this is needed
fig.savefig('out.pdf')

"""