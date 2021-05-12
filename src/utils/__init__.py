
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error, mean_absolute_error
from sklearn.utils import check_array
import matplotlib.ticker as mticker


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


"""
Plotting config
"""

# Using seaborn's style
plt.style.use('seaborn-dark-palette') 
# 'seaborn-colorblind', 'seaborn-deep', 'science'

# text rendering
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
}

plt.rcParams.update(tex_fonts)


    
def get_train_results(model_name, dset_name):
    
    predictions = np.load('output/'+model_name+'/predictions_%s.npy'%dset_name)
    y_true = np.load('output/'+model_name+'/y_test_%s.npy'%dset_name)
    hist = pd.read_pickle('output/'+model_name+'/train_hist_%s.pkl'%dset_name)
    
    return y_true, predictions, hist



"""
Plotting utils
"""


def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def adjust_spines(ax, spines, pos=10):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', pos))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)



def set_size(width=345, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def add_scatterplot(axs, ax_idx, y_test, predictions, str_label=None, nticks=None):

    err = mean_absolute_percentage_error(y_test, predictions)

    if nticks:
        axs[ax_idx].locator_params(axis='y', nbins=nticks)
        axs[ax_idx].locator_params(axis='x', nbins=nticks)

    axs[ax_idx].scatter(y_test, predictions, label='$\delta_0$', color='#595959')#, s=10.)
    axs[ax_idx].set_xlabel('True $\delta_0$ [rad]')
    axs[ax_idx].set_ylabel('Predicted $\delta_0$ [rad]')
    axs[ax_idx].text(0.1, 0.85, str_label, transform=axs[ax_idx].transAxes, size=12)
    axs[ax_idx].text(
        0.36, 0.1, "MAPE ="+" {:.2f} $\%$".format(err), transform=axs[ax_idx].transAxes, size=12)
    axs[ax_idx].axline((1, 1), slope=1, linestyle='dashed', color='k')
    axs[ax_idx].set_xlim(min(y_test), max(y_test))
    axs[ax_idx].set_ylim(min(predictions), max(predictions))
    axs[ax_idx].grid(linewidth=0.5, linestyle='--')


def add_history_plot(ax, history, str_label=None, epoch_xlim=None, epoch_ylim=None):

    ax.plot(history['loss'], label='val. set')
    ax.set_ylabel('MSE')
    ax.set_xlabel('epoch')
    ax.legend(loc='upper right', frameon=False)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    if epoch_xlim:
        ax.set_xlim(epoch_xlim)

    if epoch_ylim:
        ax.set_ylim(epoch_ylim)
    #else:
        #max_loss = max(history['loss'])
        #ax.set_ylim(0, max_loss+max_loss/100*10)

    ax.text(
        0.15, 0.9, str_label, transform=ax.transAxes, size=12)

