# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import sys
sys.path.append(os.getenv('HOME') + '/bin/python/libs')
# just in case notebook was not launched with the option
# %pylab inline

import pylab as plt
import numpy as np
from scipy import sparse
import matplotlib as mpl
from matplotlib.mlab import griddata
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
from scipy.sparse import coo_matrix
from scipy.signal import convolve2d, convolve, gaussian
from copy import deepcopy
import re

try:
    import faststats
except:
    faststats = None

# ==============================================================================
# Python 3 compatibility behavior
# ==============================================================================
# remap some python 2 built-ins on to py3k behavior or equivalent
# Most of them become generators
import operator

PY3 = sys.version_info[0] > 2

if PY3:
    iteritems = operator.methodcaller('items')
    itervalues = operator.methodcaller('values')
    basestring = (str, bytes)
else:
    range = xrange
    from itertools import izip as zip
    iteritems = operator.methodcaller('iteritems')
    itervalues = operator.methodcaller('itervalues')
    basestring = (str, unicode)


# ==============================================================================
# ============= FIGURE SETUP FUNCTIONS =========================================
# ==============================================================================
def tight_layout():
    from matplotlib import get_backend
    from pylab import gcf
    if get_backend().lower() in ['agg', 'macosx']:
        gcf().set_tight_layout(True)
    else:
        plt.tight_layout()


def theme(ax=None, minorticks=False):
    """ update plot to make it nice and uniform """
    from matplotlib.ticker import AutoMinorLocator
    from pylab import rcParams, gca, tick_params
    if minorticks:
        if ax is None:
            ax = gca()
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    tick_params(which='both', width=rcParams['lines.linewidth'])


def steppify(x, y):
    """ Steppify a curve (x,y). Useful for manually filling histograms """
    dx = 0.5 * (x[1:] + x[:-1])
    xx = np.zeros( 2 * len(dx), dtype=float)
    yy = np.zeros( 2 * len(y), dtype=float)
    xx[0::2], xx[1::2] = dx, dx
    yy[0::2], yy[1::2] = y, y
    xx = np.concatenate(([x[0] - (dx[0] - x[0])], xx, [x[-1] + (x[-1] - dx[-1])]))
    return xx, yy


def colorify(data, vmin=None, vmax=None, cmap=plt.cm.Spectral):
    """ Associate a color map to a quantity vector """
    try:
        from matplotlib.colors import Normalize
    except ImportError:
        # old mpl

        from matplotlib.colors import normalize as Normalize

    _vmin = vmin or min(data)
    _vmax = vmax or max(data)
    cNorm = Normalize(vmin=_vmin, vmax=_vmax)

    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    try:
        colors = scalarMap.to_rgba(data)
    except:
        colors = list(map(scalarMap.to_rgba, data))
    return colors, scalarMap


def devectorize_axes(ax=None, dpi=None, transparent=True):
    """Convert axes contents to a png.

    This is useful when plotting many points, as the size of the saved file
    can become very large otherwise.

    Parameters
    ----------
    ax : Axes instance (optional)
        Axes to de-vectorize.  If None, this uses the current active axes
        (plt.gca())
    dpi: int (optional)
        resolution of the png image.  If not specified, the default from
        'savefig.dpi' in rcParams will be used
    transparent : bool (optional)
        if True (default) then the PNG will be made transparent

    Returns
    -------
    ax : Axes instance
        the in-place modified Axes instance

    Examples
    --------
    The code can be used in the following way::

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = np.random.random((2, 10000))
        ax.scatter(x, y)
        devectorize_axes(ax)
        plt.savefig('devectorized.pdf')

    The resulting figure will be much smaller than the vectorized version.
    """
    from matplotlib.transforms import Bbox
    from matplotlib import image
    try:
        from io import BytesIO as StringIO
    except ImportError:
        try:
            from cStringIO import StringIO
        except ImportError:
            from StringIO import StringIO

    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    axlim = ax.axis()

    # setup: make all visible spines (axes & ticks) & text invisible
    # we need to set these back later, so we save their current state
    _sp = {}
    _txt_vis = [t.get_visible() for t in ax.texts]
    for k in ax.spines:
        _sp[k] = ax.spines[k].get_visible()
        ax.spines[k].set_visible(False)
    for t in ax.texts:
        t.set_visible(False)

    _xax = ax.xaxis.get_visible()
    _yax = ax.yaxis.get_visible()
    _patch = ax.axesPatch.get_visible()
    ax.axesPatch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # convert canvas to PNG
    extents = ax.bbox.extents / fig.dpi
    sio = StringIO()
    plt.savefig(sio, format='png', dpi=dpi,
                transparent=transparent,
                bbox_inches=Bbox([extents[:2], extents[2:]]))
    sio.seek(0)
    im = image.imread(sio)

    # clear everything on axis (but not text)
    ax.lines = []
    ax.patches = []
    ax.tables = []
    ax.artists = []
    ax.images = []
    ax.collections = []

    # Show the image
    ax.imshow(im, extent=axlim, aspect='auto', interpolation='nearest')

    # restore all the spines & text
    for k in ax.spines:
        ax.spines[k].set_visible(_sp[k])
    for t, v in zip(ax.texts, _txt_vis):
        t.set_visible(v)
    ax.axesPatch.set_visible(_patch)
    ax.xaxis.set_visible(_xax)
    ax.yaxis.set_visible(_yax)

    if plt.isinteractive():
        plt.draw()

    return ax


def hist_with_err(x, xerr, bins=None, normed=False, step=False, *kwargs):
    from scipy import integrate

    # check inputs
    assert( len(x) == len(xerr) ), 'data size mismatch'
    _x = np.asarray(x).astype(float)
    _xerr = np.asarray(xerr).astype(float)

    # def the evaluation points
    if (bins is None) | (not hasattr(bins, '__iter__')):
        m = (_x - _xerr).min()
        M = (_x + _xerr).max()
        dx = M - m
        m -= 0.2 * dx
        M += 0.2 * dx
        if bins is not None:
            N = int(bins)
        else:
            N = 10
        _xp = np.linspace(m, M, N)
    else:
        _xp = 0.5 * (bins[1:] + bins[:-1])

    def normal(v, mu, sig):
        norm_pdf = 1. / (np.sqrt(2. * np.pi) * sig ) * np.exp( - ( (v - mu ) / (2. * sig) ) ** 2 )
        return norm_pdf / integrate.simps(norm_pdf, v)

    _yp = np.array([normal(_xp, xk, xerrk) for xk, xerrk in zip(_x, _xerr) ]).sum(axis=0)

    if normed:
        _yp /= integrate.simps(_yp, _xp)

    if step:
        return steppify(_xp, _yp)
    else:
        return _xp, _yp


def hist_with_err_bootstrap(x, xerr, bins=None, normed=False, nsample=50, step=False, **kwargs):
    x0, y0 = hist_with_err(x, xerr, bins=bins, normed=normed, step=step, **kwargs)

    yn = np.empty( (nsample, len(y0)), dtype=float)
    yn[0, :] = y0
    for k in range(nsample - 1):
        idx = np.random.randint(0, len(x), len(x))
        yn[k, :] = hist_with_err(x[idx], xerr[idx], bins=bins, normed=normed, step=step, **kwargs)[1]

    return x0, yn


def __get_hesse_bins__(_x, _xerr=0., bins=None, margin=0.2):
    if (bins is None) | (not hasattr(bins, '__iter__')):
        m = (_x - _xerr).min()
        M = (_x + _xerr).max()
        dx = M - m
        m -= margin * dx
        M += margin * dx
        if bins is not None:
            N = int(bins)
        else:
            N = 10
        _xp = np.linspace(m, M, N)
    else:
        _xp = 0.5 * (bins[1:] + bins[:-1])
    return _xp


def scatter_contour(x, y,
                    levels=10,
                    bins=40,
                    threshold=50,
                    log_counts=False,
                    histogram2d_args={},
                    plot_args={},
                    contour_args={},
                    ax=None):
    """Scatter plot with contour over dense regions

    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot
    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels
    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours
    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.
    histogram2d_args : dict
        keyword arguments passed to numpy.histogram2d
        see doc string of numpy.histogram2d for more information
    plot_args : dict
        keyword arguments passed to pylab.scatter
        see doc string of pylab.scatter for more information
    contourf_args : dict
        keyword arguments passed to pylab.contourf
        see doc string of pylab.contourf for more information
    ax : pylab.Axes instance
        the axes on which to plot.  If not specified, the current
        axes will be used
    """
    if ax is None:
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, **histogram2d_args)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    i_min = np.argmin(levels)

    # draw a zero-width line: this gives us the outer polygon to
    # reduce the number of points we draw
    # somewhat hackish... we could probably get the same info from
    # the filled contour below.
    outline = ax.contour(H.T, levels[i_min:i_min + 1],
                         linewidths=0, extent=extent)
    try:
        outer_poly = outline.allsegs[0][0]

        ax.contourf(H.T, levels, extent=extent, **contour_args)
        X = np.hstack([x[:, None], y[:, None]])

        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X)
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]

        ax.plot(Xplot[:, 0], Xplot[:, 1], zorder=1, **plot_args)
    except IndexError:
        ax.plot(x, y, zorder=1, **plot_args)


def latex_float(f, precision=0.2, delimiter=r'\times'):
    float_str = ("{0:" + str(precision) + "g}").format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return (r"{0}" + delimiter + "10^{{{1}}}").format(base, int(exponent))
    else:
        return float_str


# ==============================================================================
# ==============================================================================
# ==============================================================================

def ezrc(fontSize=22., lineWidth=2., labelSize=None, tickmajorsize=10,
         tickminorsize=5, figsize=(8, 6)):
    """
    slides - Define params to make pretty fig for slides
    """
    from pylab import rc, rcParams
    if labelSize is None:
        labelSize = fontSize + 5
    rc('figure', figsize=figsize)
    rc('lines', linewidth=lineWidth)
    rcParams['grid.linewidth'] = lineWidth
    rcParams['font.sans-serif'] = ['Helvetica']
    rcParams['font.serif'] = ['Helvetica']
    rcParams['font.family'] = ['Times New Roman']
    rc('font', size=fontSize, family='serif', weight='bold')
    rc('axes', linewidth=lineWidth, labelsize=labelSize)
    rc('legend', borderpad=0.1, markerscale=1., fancybox=False)
    rc('text', usetex=True)
    rc('image', aspect='auto')
    rc('ps', useafm=True, fonttype=3)
    rcParams['xtick.major.size'] = tickmajorsize
    rcParams['xtick.minor.size'] = tickminorsize
    rcParams['ytick.major.size'] = tickmajorsize
    rcParams['ytick.minor.size'] = tickminorsize
    rcParams['text.latex.preamble'] = ["\\usepackage{amsmath}"]


def hide_axis(where, ax=None):
    ax = ax or plt.gca()
    if type(where) == str:
        _w = [where]
    else:
        _w = where
    [sk.set_color('None') for k, sk in ax.spines.items() if k in _w ]

    if 'top' in _w and 'bottom' in _w:
        ax.xaxis.set_ticks_position('none')
    elif 'top' in _w:
        ax.xaxis.set_ticks_position('bottom')
    elif 'bottom' in _w:
        ax.xaxis.set_ticks_position('top')

    if 'left' in _w and 'right' in _w:
        ax.yaxis.set_ticks_position('none')
    elif 'left' in _w:
        ax.yaxis.set_ticks_position('right')
    elif 'right' in _w:
        ax.yaxis.set_ticks_position('left')

    plt.draw_if_interactive()


def despine(fig=None, ax=None, top=True, right=True,
            left=False, bottom=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure
        figure to despine all axes of, default uses current figure
    ax : matplotlib axes
        specific axes object to despine
    top, right, left, bottom : boolean
        if True, remove that spine

    """
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            ax_i.spines[side].set_visible(not locals()[side])


def shift_axis(which, delta, where='outward', ax=None):
    ax = ax or plt.gca()
    if type(which) == str:
        _w = [which]
    else:
        _w = which

    scales = (ax.xaxis.get_scale(), ax.yaxis.get_scale())
    lbls = (ax.xaxis.get_label(), ax.yaxis.get_label())

    for wk in _w:
        ax.spines[wk].set_position((where, delta))

    ax.set_xscale(scales[0])
    ax.set_yscale(scales[1])
    ax.xaxis.set_label(lbls[0])
    ax.yaxis.set_label(lbls[1])
    plt.draw_if_interactive()


class AutoLocator(MaxNLocator):
    def __init__(self, nbins=9, steps=[1, 2, 5, 10], **kwargs):
        MaxNLocator.__init__(self, nbins=nbins, steps=steps, **kwargs )


def setMargins(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        """
        Tune the subplot layout via the meanings (and suggested defaults) are::

            left  = 0.125  # the left side of the subplots of the figure
            right = 0.9    # the right side of the subplots of the figure
            bottom = 0.1   # the bottom of the subplots of the figure
            top = 0.9      # the top of the subplots of the figure
            wspace = 0.2   # the amount of width reserved for blank space between subplots
            hspace = 0.2   # the amount of height reserved for white space between subplots

        The actual defaults are controlled by the rc file

        """
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        plt.draw_if_interactive()


def setNmajors(xval=None, yval=None, ax=None, mode='auto', **kwargs):
        """
        setNmajors - set major tick number
        see figure.MaxNLocator for kwargs
        """
        if ax is None:
                ax = plt.gca()
        if (mode == 'fixed'):
                if xval is not None:
                        ax.xaxis.set_major_locator(MaxNLocator(xval, **kwargs))
                if yval is not None:
                        ax.yaxis.set_major_locator(MaxNLocator(yval, **kwargs))
        elif (mode == 'auto'):
                if xval is not None:
                        ax.xaxis.set_major_locator(AutoLocator(xval, **kwargs))
                if yval is not None:
                        ax.yaxis.set_major_locator(AutoLocator(yval, **kwargs))

        plt.draw_if_interactive()


def crazy_histogram2d(x, y, bins=10, weights=None, reduce_w=None, NULL=None, reinterp=None):
    """
    Compute the sparse bi-dimensional histogram of two data samples where *x*,
    and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).

    If *weights* is specified, it specifies values at the coordinate (x[i],
    y[i]). These values are accumulated for each bin and then reduced according
    to *reduce_w* function, which defaults to numpy's sum function (np.sum).
    (If *weights* is specified, it must also be a 1-D sequence of the same
    length as *x* and *y*.)

    INPUTS:
        x       ndarray[ndim=1]         first data sample coordinates
        y       ndarray[ndim=1]         second data sample coordinates

    KEYWORDS:
        bins                            the bin specification
                   int                     the number of bins for the two dimensions (nx=ny=bins)
                or [int, int]              the number of bins in each dimension (nx, ny = bins)
        weights     ndarray[ndim=1]     values *w_i* weighing each sample *(x_i, y_i)*
                                        accumulated and reduced (using reduced_w) per bin
        reduce_w    callable            function that will reduce the *weights* values accumulated per bin
                                        defaults to numpy's sum function (np.sum)
        NULL        value type          filling missing data value
        reinterp    str                 values are [None, 'nn', linear']
                                        if set, reinterpolation is made using mlab.griddata to fill missing data
                                        within the convex polygone that encloses the data

    OUTPUTS:
        B           ndarray[ndim=2]     bi-dimensional histogram
        extent      tuple(4)            (xmin, xmax, ymin, ymax) entension of the histogram
        steps       tuple(2)            (dx, dy) bin size in x and y direction

    """
    # define the bins (do anything you want here but needs edges and sizes of the 2d bins)
    try:
        nx, ny = bins
    except TypeError:
        nx = ny = bins

    # values you want to be reported
    if weights is None:
        weights = np.ones(x.size)

    if reduce_w is None:
        reduce_w = np.sum
    else:
        if not hasattr(reduce_w, '__call__'):
            raise TypeError('reduce function is not callable')

    # culling nans
    finite_inds = (np.isfinite(x) & np.isfinite(y) & np.isfinite(weights))
    _x = np.asarray(x)[finite_inds]
    _y = np.asarray(y)[finite_inds]
    _w = np.asarray(weights)[finite_inds]

    if not (len(_x) == len(_y)) & (len(_y) == len(_w)):
        raise ValueError('Shape mismatch between x, y, and weights: {}, {}, {}'.format(_x.shape, _y.shape, _w.shape))

    xmin, xmax = _x.min(), _x.max()
    ymin, ymax = _y.min(), _y.max()
    dx = (xmax - xmin) / (nx - 1.0)
    dy = (ymax - ymin) / (ny - 1.0)

    # Basically, this is just doing what np.digitize does with one less copy
    xyi = np.vstack((_x, _y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # xyi contains the bins of each point as a 2d array [(xi,yi)]

    d = {}
    for e, k in enumerate(xyi.T):
        key = (k[0], k[1])

        if key in d:
            d[key].append(_w[e])
        else:
            d[key] = [_w[e]]

    _xyi = np.array(d.keys()).T
    _w   = np.array([ reduce_w(v) for v in d.values() ])

    # exploit a sparse coo_matrix to build the 2D histogram...
    _grid = sparse.coo_matrix((_w, _xyi), shape=(nx, ny))

    if reinterp is None:
        # convert sparse to array with filled value
        #  grid.toarray() does not account for filled value
        #  sparse.coo.coo_todense() does actually add the values to the existing ones, i.e. not what we want -> brute force
        if NULL is None:
            B = _grid.toarray()
        else:  # Brute force only went needed
            B = np.zeros(_grid.shape, dtype=_grid.dtype)
            B.fill(NULL)
            for (x, y, v) in zip(_grid.col, _grid.row, _grid.data):
                B[y, x] = v
    else:  # reinterp
        xi = np.arange(nx, dtype=float)
        yi = np.arange(ny, dtype=float)
        B = griddata(_grid.col.astype(float), _grid.row.astype(float), _grid.data, xi, yi, interp=reinterp)

    return B, (xmin, xmax, ymin, ymax), (dx, dy)


def histplot(data, bins=10, range=None, normed=False, weights=None, density=None, ax=None, **kwargs):
    """ plot an histogram of data `a la R`: only bottom and left axis, with
    dots at the bottom to represent the sample

    Example
    -------
        import numpy as np
        x = np.random.normal(0, 1, 1e3)
        histplot(x, bins=50, density=True, ls='steps-mid')
    """
    h, b = np.histogram(data, bins, range, normed, weights, density)
    if ax is None:
        ax = plt.gca()
    x = 0.5 * (b[:-1] + b[1:])
    l = ax.plot(x, h, **kwargs)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    _w = ['top', 'right']
    [ ax.spines[side].set_visible(False) for side in _w ]

    for wk in ['bottom', 'left']:
        ax.spines[wk].set_position(('outward', 10))

    ylim = ax.get_ylim()
    ax.plot(data, -0.02 * max(ylim) * np.ones(len(data)), '|', color=l[0].get_color())
    ax.set_ylim(-0.02 * max(ylim), max(ylim))


def scatter_plot(x, y, ellipse=False, levels=[0.99, 0.95, 0.68], color='w', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if faststats is not None:
        im, e = faststats.fastkde.fastkde(x, y, (50, 50), adjust=2.)
        V = im.max() * np.asarray(levels)

        plt.contour(im.T, levels=V, origin='lower', extent=e, linewidths=[1, 2, 3], colors=color)

    ax.plot(x, y, 'b,', alpha=0.3, zorder=-1, rasterized=True)

    if ellipse is True:
        data = np.vstack([x, y])
        mu = np.mean(data, axis=1)
        cov = np.cov(data)
        error_ellipse(mu, cov, ax=plt.gca(), edgecolor="g", ls="dashed", lw=4, zorder=2)


def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
                          width=2 * np.sqrt(S[0]) * factor,
                          height=2 * np.sqrt(S[1]) * factor,
                          angle=theta,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot


def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    # -----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    # -----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    # -----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    # -----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]


def quantiles(x, qlist=[2.5, 25, 50, 75, 97.5]):
    """computes quantiles from an array

    Quantiles :=  points taken at regular intervals from the cumulative
    distribution function (CDF) of a random variable. Dividing ordered data
    into q essentially equal-sized data subsets is the motivation for
    q-quantiles; the quantiles are the data values marking the boundaries
    between consecutive subsets.

    The quantile with a fraction 50 is called the median
    (50% of the distribution)

    Inputs:
        x     - variable to evaluate from
        qlist - quantiles fraction to estimate (in %)

    Outputs:
        Returns a dictionary of requested quantiles from array
    """
    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort, then transpose back
        sx = np.transpose(np.sort(np.transpose(x)))
    else:
        # Sort univariate node
        sx = np.sort(x)

    try:
        # Generate specified quantiles
        quants = [sx[int(len(sx) * q / 100.0)] for q in qlist]

        return dict(zip(qlist, quants))

    except IndexError:
        print("Too few elements for quantile calculation")


def get_optbins(data, method='freedman', ret='N'):
    """ Determine the optimal binning of the data based on common estimators
    and returns either the number of bins of the width to use.

    input:
        data    1d dataset to estimate from
    keywords:
        method  the method to use: str in {sturge, scott, freedman}
        ret set to N will return the number of bins / edges
            set to W will return the width
    refs:
        Sturges, H. A. (1926)."The choice of a class interval". J. American Statistical Association, 65-66
        Scott, David W. (1979), "On optimal and data-based histograms". Biometrika, 66, 605-610
        Freedman, D.; Diaconis, P. (1981). "On the histogram as a density estimator: L2 theory".
                Zeitschrift fur Wahrscheinlichkeitstheorie und verwandte Gebiete, 57, 453-476
        Scargle, J.D. et al (2012) "Studies in Astronomical Time Series Analysis. VI. Bayesian
        Block Representations."
    """
    x = np.asarray(data)
    n = x.size
    r = x.max() - x.min()

    def sturge():
        if (n <= 30):
            print("Warning: Sturge estimator can perform poorly for small samples")
        k = int(np.log(n) + 1)
        h = r / k
        return h, k

    def scott():
        h = 3.5 * np.std(x) * float(n) ** (-1. / 3.)
        k = int(r / h)
        return h, k

    def freedman():
        q = quantiles(x, [25, 75])
        h = 2 * (q[75] - q[25]) * float(n) ** (-1. / 3.)
        k = int(r / h)
        return h, k

    def bayesian():
        r = bayesian_blocks(x)
        return np.diff(r),r

    m = {'sturge': sturge, 'scott': scott, 'freedman': freedman,
         'bayesian': bayesian}

    if method.lower() in m:
        s = m[method.lower()]()
        if ret.lower() == 'n':
            return s[1]
        elif ret.lower() == 'w':
            return s[0]
    else:
        return None


def plotMAP(x, ax=None, error=0.01, frac=[0.65,0.95, 0.975], usehpd=True,
            hist={'histtype':'step'}, vlines={}, fill={},
            optbins={'method':'freedman'}, *args, **kwargs):
    """ Plot the MAP of a given sample and add statistical info
    If not specified, binning is assumed from the error value or using
    mystats.optbins if available.
    if mystats module is not available, hpd keyword has no effect

    inputs:
        x   dataset
    keywords
        ax  axe object to use during plotting
        error   error to consider on the estimations
        frac    fractions of sample to highlight (def 65%, 95%, 97.5%)
        hpd if set, uses mystats.hpd to estimate the confidence intervals

        hist    keywords forwarded to hist command
        optbins keywords forwarded to mystats.optbins command
        vlines  keywords forwarded to vlines command
        fill    keywords forwarded to fill command
        """
    _x = np.ravel(x)
    if ax is None:
        ax = plt.gca()
    if not ('bins' in hist):
        bins = get_optbins(x, method=optbins['method'], ret='N')
        n, b, p = ax.hist(_x, bins=bins, *args, **hist)
    else:
        n, b, p = ax.hist(_x, *args, **hist)
    c = 0.5 * (b[:-1] + b[1:])
    # dc = 0.5 * (b[:-1] - b[1:])
    ind = n.argmax()
    _ylim = ax.get_ylim()
    if usehpd is True:
        _hpd = hpd(_x, 1 - 0.01)
        ax.vlines(_hpd, _ylim[0], _ylim[1], **vlines)
        for k in frac:
            nx = hpd(_x, 1. - k)
            ax.fill_between(nx, _ylim[0], _ylim[1], alpha=0.4 / float(len(frac)), zorder=-1, **fill)
    else:
        ax.vlines(c[ind], _ylim[0], _ylim[1], **vlines)
        cx = c[ n.argsort() ][::-1]
        cn = n[ n.argsort() ][::-1].cumsum()
        for k in frac:
            sx = cx[np.where(cn <= cn[-1] * float(k))]
            sx = [sx.min(), sx.max()]
            ax.fill_between(sx, _ylim[0], _ylim[1], alpha=0.4 / float(len(frac)), zorder=-1, **fill)
    theme(ax=ax)
    ax.set_xlabel(r'Values')
    ax.set_ylabel(r'Counts')


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width"""

    # Initialize interval
    min_int = [None, None]

    try:

        # Number of elements in trace
        n = len(x)

        # Start at far left
        start, end = 0, int(n * (1 - alpha))

        # Initialize minimum width to large value
        min_width = np.inf

        while end < n:

            # Endpoints of interval
            hi, lo = x[end], x[start]

            # Width of interval
            width = hi - lo

            # Check to see if width is narrower than minimum
            if width < min_width:
                min_width = width
                min_int = [lo, hi]

            # Increment endpoints
            start += 1
            end += 1

        return min_int

    except IndexError:
        print('Too few elements for interval calculation')
        return [None, None]


def getPercentileLevels(h, frac=[0.5, 0.65, 0.95, 0.975]):
    """
    Return image levels that corresponds to given percentiles values
    Uses the cumulative distribution of the sorted image density values
    Hence this works also for any nd-arrays
    inputs:
        h   array
    outputs:
        res array containing level values
    keywords:
        frac    sample fractions (percentiles)
            could be scalar or iterable
            default: 50%, 65%, 95%, and 97.5%

    """
    if getattr(frac, '__iter__', False):
        return np.asarray( [getPercentileLevels(h, fk) for fk in frac])

    if not ((frac >= 0.) & (frac < 1.)):
        raise ValueError("Expecting a sample fraction in 'frac' and got %f" % frac)

    # flatten the array to a 1d list
    val = h.ravel()
    # inplace sort
    val.sort()
    # reverse order
    rval = val[::-1]
    # cumulative values
    cval = rval.cumsum()
    cval = (cval - cval[0]) / (cval[-1] - cval[0])
    # retrieve the largest indice up to the fraction of the sample we want
    ind = np.where(cval <= cval[-1] * float(frac))[0].max()
    res = rval[ind]
    del val, cval, ind, rval
    return res


def fastkde(x, y, gridsize=(200, 200), extents=None, nocorrelation=False,
            weights=None, adjust=1.):
    """
    A fft-based Gaussian kernel density estimate (KDE)
    for computing the KDE on a regular grid

    Note that this is a different use case than scipy's original
    scipy.stats.kde.gaussian_kde

    IMPLEMENTATION
    --------------

    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    It computes the sparse bi-dimensional histogram of two data samples where
    *x*, and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).
    histogram of the data is a faster implementation than numpy.histogram as it
    avoids intermediate copies and excessive memory usage!


    This function is typically *several orders of magnitude faster* than
    scipy.stats.kde.gaussian_kde.  For large (>1e7) numbers of points, it
    produces an essentially identical result.

    Boundary conditions on the data is corrected by using a symmetric /
    reflection condition. Hence the limits of the dataset does not affect the
    pdf estimate.

    Parameters
    ----------

        x, y:  ndarray[ndim=1]
            The x-coords, y-coords of the input data points respectively

        gridsize: tuple
            A (nx,ny) tuple of the size of the output grid (default: 200x200)

        extents: (xmin, xmax, ymin, ymax) tuple
            tuple of the extents of output grid (default: extent of input data)

        nocorrelation: bool
            If True, the correlation between the x and y coords will be ignored
            when preforming the KDE. (default: False)

        weights: ndarray[ndim=1]
            An array of the same shape as x & y that weights each sample (x_i,
            y_i) by each value in weights (w_i).  Defaults to an array of ones
            the same size as x & y. (default: None)

        adjust : float
            An adjustment factor for the bw. Bandwidth becomes bw * adjust.

    Returns
    -------
        g: ndarray[ndim=2]
            A gridded 2D kernel density estimate of the input points.

        e: (xmin, xmax, ymin, ymax) tuple
            Extents of g

    """
    # Variable check
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)

    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size as input x & y arrays!')

    # Optimize gridsize ------------------------------------------------------
    # Make grid and discretize the data and round it to the next power of 2
    # to optimize with the fft usage
    if gridsize is None:
        gridsize = np.asarray([np.max((len(x), 512.)), np.max((len(y), 512.))])
    gridsize = 2 ** np.ceil(np.log2(gridsize))  # round to next power of 2

    nx, ny = gridsize

    # Make the sparse 2d-histogram -------------------------------------------
    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Basically, this is just doing what np.digitize does with one less copy
    # xyi contains the bins of each point as a 2d array [(xi,yi)]
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y.
    # Exploit a sparse coo_matrix avoiding np.histogram2d due to excessive
    # memory usage with many points
    grid = coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Kernel Preliminary Calculations ---------------------------------------
    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = n ** (-1.0 / 6.) * adjust  # For 2D

    # Make the gaussian kernel ---------------------------------------------

    # First, determine the bandwidth using Scott's rule
    # (note that Silvermann's rule gives the # same value for 2d datasets)
    std_devs = np.sqrt(np.diag(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.sum(kernel, axis=0) / 2.0
    kernel = np.exp(-kernel)
    kernel = kernel.reshape((kern_ny, kern_nx))

    # Convolve the histogram with the gaussian kernel
    # use boundary=symm to correct for data boundaries in the kde
    grid = convolve2d(grid, kernel, mode='same', boundary='symm')

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * cov * scotts_factor ** 2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    return grid, (xmin, xmax, ymin, ymax), dx, dy


def percentile(data, percentiles, weights=None):
    """Compute weighted percentiles.

    If the weights are equal, this is the same as normal percentiles.
    Elements of the data and wt arrays correspond to each other and must have
    equal length.
    If wt is None, this function calls numpy's percentile instead (faster)


    Implementation
    --------------

    The method implemented here extends the commom percentile estimation method
    (linear interpolation beteeen closest ranks) approach in a natural way.
    Suppose we have positive weights, W= [W_i], associated, respectively, with
    our N sorted sample values, D=[d_i]. Let S_n = Sum_i=0..n {w_i} the
    the n-th partial sum of the weights. Then the n-th percentile value is
    given by the interpolation between its closest values v_k, v_{k+1}:

        v = v_k + (p - p_k) / (p_{k+1} - p_k) * (v_{k+1} - v_k)

    where
        p_n = 100/S_n * (S_n - w_n/2)

    Note that the 50th weighted percentile is known as the weighted median.


    Parameters
    ----------
    data: ndarray[float, ndim=1]
        data points

    percentiles: ndarray[float, ndim=1]
        percentiles to use. (between 0 and 100)

    weights: ndarray[float, ndim=1] or None
        Weights of each point in data
        All the weights must be non-negative and the sum must be
        greater than zero.

    Returns
    -------
    val: ndarray
        the weighted percentiles of the data.
    """
    # check if actually weighted percentiles is needed
    if weights is None:
        return np.percentile(data, list(percentiles))
    if np.equal(weights, 1.).all():
        return np.percentile(data, list(percentiles))

    # make sure percentiles are fractions between 0 and 1
    if not np.greater_equal(percentiles, 0.0).all():
        raise ValueError("Percentiles less than 0")
    if not np.less_equal(percentiles, 100.0).all():
        raise ValueError("Percentiles greater than 100")

    # Make sure data is in correct shape
    shape = np.shape(data)
    n = len(data)
    if (len(shape) != 1):
        raise ValueError("wrong data shape, expecting 1d")

    if len(weights) != n:
        print(n, len(weights))
        raise ValueError("weights must be the same shape as data")
    if not np.greater_equal(weights, 0.0).all():
        raise ValueError("Not all weights are non-negative.")

    _data = np.asarray(data, dtype=float)

    if hasattr(percentiles, '__iter__'):
        _p = np.asarray(percentiles, dtype=float) * 0.01
    else:
        _p = np.asarray([percentiles * 0.01], dtype=float)

    _wt = np.asarray(weights, dtype=float)

    len_p = len(_p)
    sd = np.empty(n, dtype=float)
    sw = np.empty(n, dtype=float)
    aw = np.empty(n, dtype=float)
    o = np.empty(len_p, dtype=float)

    i = np.argsort(_data)
    np.take(_data, i, axis=0, out=sd)
    np.take(_wt, i, axis=0, out=sw)
    np.add.accumulate(sw, out=aw)

    if not aw[-1] > 0:
        raise ValueError("Nonpositive weight sum")

    w = (aw - 0.5 * sw) / aw[-1]

    spots = np.searchsorted(w, _p)
    for (pk, s, p) in zip(range(len_p), spots, _p):
        if s == 0:
            o[pk] = sd[0]
        elif s == n:
            o[pk] = sd[n - 1]
        else:
            f1 = (w[s] - p) / (w[s] - w[s - 1])
            f2 = (p - w[s - 1]) / (w[s] - w[s - 1])
            o[pk] = sd[s - 1] * f1 + sd[s] * f2
    return o


class KDE_2d(object):
    def __init__(self, x, y, gridsize=(100, 100), extents=None,
                 nocorrelation=False, weights=None, adjust=0.5):

        im, e, dx, dy = fastkde(x, y, gridsize=gridsize, extents=extents,
                                nocorrelation=nocorrelation, weights=weights,
                                adjust=adjust)
        self.x = x
        self.y = y
        self.im = im
        self.e = e
        self.dx = dx
        self.dy = dy

    @property
    def peak(self):
        im = self.im
        e = self.e
        dx = self.dx
        dy = self.dy
        best_idx = (im.argmax() / im.shape[1], im.argmax() % im.shape[1])
        best = (best_idx[0] * dx + e[0], best_idx[1] * dy + e[2])
        return best

    def nice_levels(self, N=5):
        """ Generates N sigma levels from a image map

        Parameters
        ----------

        H: ndarray
            values to find levels from

        N: int
            number of sigma levels to find

        Returns
        -------
        lvls: sequence
            Levels corresponding to 1..(N + 1) sigma levels
        """
        V = 1.0 - np.exp(-0.5 * np.arange(0.5, 0.5 * (N + 1 + 0.1), 0.5) ** 2)
        Hflat = self.im.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]

        for i, v0 in enumerate(V):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        return np.sort(V)

    def percentiles_lvl(self, frac):
        return getPercentileLevels(self.im, frac=frac)

    def imshow(self, **kwargs):
        defaults = {'origin': 'lower', 'cmap': plt.cm.Greys,
                    'interpolation':'nearest', 'aspect':'auto'}
        defaults.update(**kwargs)
        ax = kwargs.pop('ax', plt.gca())
        return ax.imshow(self.im.T, extent=self.e, **defaults)

    def contour(self, *args, **kwargs):
        defaults = {'origin': 'lower', 'cmap': plt.cm.Greys,
                    'levels': np.sort(self.nice_levels())}
        defaults.update(**kwargs)
        ax = kwargs.pop('ax', plt.gca())
        return ax.contour(self.im.T, *args, extent=self.e, **defaults)

    def contourf(self, *args, **kwargs):
        defaults = {'origin': 'lower', 'cmap': plt.cm.Greys,
                    'levels': self.nice_levels()}
        defaults.update(**kwargs)
        ax = kwargs.pop('ax', plt.gca())
        return ax.contourf(self.im.T, *args,  extent=self.e, **defaults)

    def scatter(self, lvl=None, **kwargs):
        defaults = {'c': '0.0', 'color':'k', 'facecolor':'k', 'edgecolor':'None'}
        defaults.update(**kwargs)

        xe = self.e[0] + self.dx * np.arange(0, self.im.shape[1])
        ye = self.e[2] + self.dy * np.arange(0, self.im.shape[0])
        x = self.x
        y = self.y

        if lvl is not None:
            nx = np.ceil(np.interp(x, 0.5 * (xe[:-1] + xe[1:]), range(len(xe) - 1)))
            ny = np.ceil(np.interp(y, 0.5 * (ye[:-1] + ye[1:]), range(len(ye) - 1)))
            nh = [ self.im[nx[k], ny[k]] for k in range(len(x)) ]
            ind = np.where(nh < np.min(lvl))
            plt.scatter(x[ind], y[ind], **kwargs)
        else:
            plt.scatter(x, y, **kwargs)

    def plot(self, contour={}, scatter={}, **kwargs):
        # levels = np.linspace(self.im.min(), self.im.max(), 10)[1:]
        levels = self.nice_levels()
        c_defaults = {'origin': 'lower', 'cmap': plt.cm.Greys_r, 'levels':
                      levels}
        c_defaults.update(**contour)

        c = self.contourf(**c_defaults)

        lvls = np.sort(c.levels)
        s_defaults = {'c': '0.0', 'edgecolor':'None', 's':2}
        s_defaults.update(**scatter)

        self.scatter(lvl=[lvls], **s_defaults)


def plot_kde2d(x, y, gridsize=(100, 100), extents=None, nocorrelation=False,
               weights=None, adjust=0.3, **kwargs):

    kde = KDE_2d(x, y, gridsize=gridsize, extents=extents,
                 nocorrelation=nocorrelation, weights=weights, adjust=adjust)

    kde.plot(**kwargs)


def fastkde1D(xin, gridsize=200, extents=None, weights=None, adjust=1.):
    """
    A fft-based Gaussian kernel density estimate (KDE)
    for computing the KDE on a regular grid

    Note that this is a different use case than scipy's original
    scipy.stats.kde.gaussian_kde

    IMPLEMENTATION
    --------------

    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    It computes the sparse bi-dimensional histogram of two data samples where
    *x*, and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).
    histogram of the data is a faster implementation than numpy.histogram as it
    avoids intermediate copies and excessive memory usage!


    This function is typically *several orders of magnitude faster* than
    scipy.stats.kde.gaussian_kde.  For large (>1e7) numbers of points, it
    produces an essentially identical result.

    **Example usage and timing**

        from scipy.stats import gaussian_kde

        def npkde(x, xe):
            kde = gaussian_kde(x)
            r = kde.evaluate(xe)
            return r
        x = np.random.normal(0, 1, 1e6)

        %timeit fastkde1D(x)
        10 loops, best of 3: 31.9 ms per loop

        %timeit npkde(x, xe)
        1 loops, best of 3: 11.8 s per loop

        ~ 1e4 speed up!!! However gaussian_kde is not optimized for this application

    Boundary conditions on the data is corrected by using a symmetric /
    reflection condition. Hence the limits of the dataset does not affect the
    pdf estimate.

    INPUTS
    ------

        xin:  ndarray[ndim=1]
            The x-coords, y-coords of the input data points respectively

        gridsize: int
            A nx integer of the size of the output grid (default: 200x200)

        extents: (xmin, xmax) tuple
            tuple of the extents of output grid (default: extent of input data)

        weights: ndarray[ndim=1]
            An array of the same shape as x that weights each sample x_i
            by w_i.  Defaults to an array of ones the same size as x (default: None)

        adjust : float
            An adjustment factor for the bw. Bandwidth becomes bw * adjust.

    OUTPUTS
    -------
        g: ndarray[ndim=2]
            A gridded 2D kernel density estimate of the input points.

        e: (xmin, xmax, ymin, ymax) tuple
            Extents of g

    """
    # Variable check
    x = np.squeeze(np.asarray(xin))

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = map(float, extents)
        x = x[ (x <= xmax) & (x >= xmin) ]

    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size as input x & y arrays!')

    # Optimize gridsize ------------------------------------------------------
    # Make grid and discretize the data and round it to the next power of 2
    # to optimize with the fft usage
    if gridsize is None:
        gridsize = np.max((len(x), 512.))
    gridsize = 2 ** np.ceil(np.log2(gridsize))  # round to next power of 2

    nx = gridsize

    # Make the sparse 2d-histogram -------------------------------------------
    dx = (xmax - xmin) / (nx - 1)

    # Basically, this is just doing what np.digitize does with one less copy
    # xyi contains the bins of each point as a 2d array [(xi,yi)]
    xyi = x - xmin
    xyi /= dx
    xyi = np.floor(xyi, xyi)
    xyi = np.vstack((xyi, np.zeros(n, dtype=int)))

    # Next, make a 2D histogram of x & y.
    # Exploit a sparse coo_matrix avoiding np.histogram2d due to excessive
    # memory usage with many points
    grid = coo_matrix((weights, xyi), shape=(nx, 1)).toarray()

    # Kernel Preliminary Calculations ---------------------------------------
    std_x = np.std(xyi[0])

    # Scaling factor for bandwidth
    scotts_factor = n ** (-1. / 5.) * adjust  # For n ** (-1. / (d + 4))

    # Silvermann n * (d + 2) / 4.)**(-1. / (d + 4)).

    # Make the gaussian kernel ---------------------------------------------

    # First, determine the bandwidth using Scott's rule
    # (note that Silvermann's rule gives the # same value for 2d datasets)
    kern_nx = np.round(scotts_factor * 2 * np.pi * std_x)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.reshape(gaussian(kern_nx, scotts_factor * std_x), (kern_nx, 1))

    # ---- Produce the kernel density estimate --------------------------------

    # Convolve the histogram with the gaussian kernel
    # use symmetric padding to correct for data boundaries in the kde
    npad = np.min((nx, 2 * kern_nx))
    grid = np.vstack( [grid[npad: 0: -1], grid, grid[nx: nx - npad: -1]] )
    grid = convolve(grid, kernel, mode='same')[npad: npad + nx]

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * std_x * std_x * scotts_factor ** 2
    norm_factor = n * dx * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    return np.squeeze(grid), (xmin, xmax), dx


class KDE_1d(object):
    def __init__(self,x, gridsize=200, extents=None, weights=None, adjust=1.):

        im, e, dx = fastkde1D(x,gridsize=gridsize, extents=extents,
                              weights=weights, adjust=adjust)
        self.x = x
        self.dx = dx
        self.im = im
        self.e = e

    @property
    def peak(self):
        im = self.im
        e = self.e
        dx = self.dx
        best = im.argmax() * dx + e[0]
        return best

    def add_markers(self, ax=None, where=0.0, orientation='horizontal',
                    jitter=0, **kwargs):

        if ax is None:
            ax = plt.gca()

        # draw the positions
        if 'marker' not in kwargs:
            if orientation == 'horizontal':
                kwargs['marker'] = '|'
            else:
                kwargs['marker'] = '_'

        if ('facecolor' not in kwargs.keys()) | ('fc' not in kwargs.keys()) | \
           ('markerfacecolor' not in kwargs.keys()) | ('mfc' not in kwargs.keys()):
            kwargs['markerfacecolor'] = 'None'
        if ('edgecolor' not in kwargs.keys()) | ('ec' not in kwargs.keys()) | \
           ('markeredgecolor' not in kwargs.keys()) | ('mec' not in kwargs.keys()):
            kwargs['markeredgecolor'] = 'k'
        if ('linestyle' not in kwargs.keys()) | ('ls' not in kwargs.keys()):
            kwargs['linestyle'] = 'None'
        if ('size' not in kwargs.keys()) | ('markersize' not in kwargs.keys()):
            kwargs['markersize'] = 3

        if orientation == 'horizontal':
            # Draw the lines
            if jitter > 0:
                pos = np.random.uniform(low=float(where - jitter),
                                        high=float(where + jitter),
                                        size=len(self.x))
                ax.plot(self.x, pos, **kwargs)
            else:
                ax.plot(self.x, float(where) * np.ones(len(self.x)), **kwargs)

            plt.draw_if_interactive()

        elif orientation == 'vertical':
            # Draw the lines
            if jitter > 0.:
                pos = np.random.uniform(low=float(where - jitter),
                                        high=float(where + jitter),
                                        size=len(self.x))
                ax.plot(pos, self.x, **kwargs)
            else:
                ax.plot(float(where) * np.ones(len(self.x)), self.x, marker='_',
                        **kwargs)

        plt.draw_if_interactive()

    def plot(self, ax=None, orientation='horizontal', cutoff=False, log=False,
             cutoff_type='std', cutoff_val=1.5, pos=100, pos_marker='line',
             pos_width=0.05, pos_kwargs={}, **kwargs):

        if ax is None:
            ax = plt.gca()

        # Draw the violin.
        if ('facecolor' not in kwargs) | ('fc' not in kwargs):
            kwargs['facecolor'] = 'y'
        if ('edgecolor' not in kwargs) | ('ec' not in kwargs):
            kwargs['edgecolor'] = 'k'
        if ('alpha' not in kwargs.keys()):
            kwargs['alpha'] = 0.5

        if 'color' in kwargs:
            kwargs['edgecolor'] = kwargs['color']
            kwargs['facecolor'] = kwargs['color']

        # Kernel density estimate for data at this position.
        violin, e = self.im, self.e
        xvals = np.linspace(e[0], e[1], len(violin))

        xvals = np.hstack(([xvals[0]], xvals, [xvals[-1]]))
        violin = np.hstack(([0], violin, [0]))

        if orientation == 'horizontal':
            ax.fill(xvals, violin, **kwargs)
        elif orientation == 'vertical':
            ax.fill_betweenx(xvals, 0, violin, **kwargs)

        plt.draw_if_interactive()


def plot_kde1d(x, gridsize=200, extents=None, weights=None, adjust=1.,
               **kwargs):
    return KDE_1d(x, gridsize=gridsize, extents=extents, weights=weights,
                  adjust=adjust).plot(**kwargs)


def plotDensity(x,y, bins=100, ax=None, Nlevels=None, levels=None,
                frac=None,
                contour={'colors':'0.0', 'linewidths':0.5},
                contourf={'cmap': plt.cm.Greys_r},
                scatter={'c':'0.0', 's':0.5, 'edgecolor':'None'},
                *args, **kwargs ):
    """
    Plot a the density of x,y given certain contour paramters and includes
    individual points (not represented by contours)

    inputs:
        x,y data to plot

    keywords:
        bins    bin definition for the density histogram
        ax  use a specific axis
        Nlevels the number of levels to use with contour
        levels  levels
        frac    percentiles to contour if specified

        Extra keywords:
        *args, **kwargs forwarded to histogram2d
        **contour       forwarded to contour function
        **contourf      forwarded to contourf function
        **plot          forwarded to contourf function

    """
    if ax is None:
        ax = plt.gca()

    if 'bins' not in kwargs:
        kwargs['bins'] = bins

    h, xe, ye = np.histogram2d(x, y, *args, **kwargs)

    if (Nlevels is None) & (levels is None) & (frac is None):
        levels = np.sort(getPercentileLevels(h))
    elif (Nlevels is not None) & (levels is None) & (frac is None):
        levels = np.linspace(2., h.max(), Nlevels)[1:].tolist() + [h.max()]
    elif (frac is not None):
        levels = getPercentileLevels(h, frac=frac)

    if not getattr(levels, '__iter__', False):
        raise AttributeError("Expecting levels variable to be iterable")

    if levels[-1] != h.max():
        levels = list(levels) + [h.max()]

    if isinstance(contourf, dict):
        cont = ax.contourf(h.T, extent=[xe[0],xe[-1], ye[0],ye[-1]],
                           levels=levels, **contourf)
    else:
        cont = None

    if isinstance(contour, dict):
        ax.contour(h.T, extent=[xe[0],xe[-1], ye[0],ye[-1]], levels=levels,
                   **contour)

    ind = np.asarray([False] * len(x))

    if cont is not None:
        nx = np.ceil(np.interp(x, 0.5 * (xe[:-1] + xe[1:]), range(len(xe) - 1)))
        ny = np.ceil(np.interp(y, 0.5 * (ye[:-1] + ye[1:]), range(len(ye) - 1)))
        nh = [ h[nx[k],ny[k]] for k in range(len(x)) ]
        ind = np.where(nh < np.min(levels))
        ax.scatter(x[ind], y[ind], **scatter)
    else:
        ax.plot(x, y, **scatter)


def make_indices(dimensions):
    """ Generates complete set of indices for given dimensions """

    level = len(dimensions)

    if level == 1:
        return range(dimensions[0])

    indices = [[]]

    while level:

        _indices = []

        for j in range(dimensions[level - 1]):

            _indices += [[j] + i for i in indices]

        indices = _indices

        level -= 1

    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices


def hpd(x, alpha):
    """Calculate HPD (minimum width BCI) of array for given alpha"""

    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, range(x.ndim)[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)


def plotCorr(l, pars, plotfunc=None, lbls=None, limits=None, triangle='lower',
             devectorize=False, *args, **kwargs):
        """ Plot correlation matrix between variables
        inputs
        -------
        l: dict
            dictionary of variables (could be a Table)

        pars: sequence of str
            parameters to use

        plotfunc: callable
            function to be called when doing the scatter plots

        lbls: sequence of str
            sequence of string to use instead of dictionary keys

        limits: dict
            impose limits for some paramters. Each limit should be pairs of values.
            No need to define each parameter limits

        triangle: str in ['upper', 'lower']
            Which side of the triangle to use.

        devectorize: bool
            if set, rasterize the figure to reduce its size

        *args, **kwargs are forwarded to the plot function

        Example
        -------
            import numpy as np
            figrc.ezrc(16, 1, 16, 5)

            d = {}

            for k in range(4):
                d[k] = np.random.normal(0, k+1, 1e4)

            plt.figure(figsize=(8 * 1.5, 7 * 1.5))
            plotCorr(d, d.keys(), plotfunc=figrc.scatter_plot)
            #plotCorr(d, d.keys(), alpha=0.2)
        """

        if lbls is None:
                lbls = pars

        if limits is None:
            limits = {}

        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}
        if not len(pars) - 1 in fontmap:
                fontmap[len(pars) - 1] = 3

        k = 1
        axes = np.empty((len(pars) + 1, len(pars)), dtype=object)
        for j in range(len(pars)):
                for i in range(len(pars)):
                        if j > i:
                                sharex = axes[j - 1, i]
                        else:
                                sharex = None

                        if i == j:
                            # Plot the histograms.
                            ax = plt.subplot(len(pars), len(pars), k)
                            axes[j, i] = ax
                            data = l[pars[i]]
                            n, b, p = ax.hist(data, bins=50, histtype="step", color=kwargs.get("color", "k"))
                            if triangle == 'upper':
                                ax.set_xlabel(lbls[i])
                                ax.set_ylabel(lbls[i])
                                ax.xaxis.set_ticks_position('bottom')
                                ax.yaxis.set_ticks_position('none')
                            else:
                                ax.yaxis.set_ticks_position('none')
                                ax.xaxis.set_ticks_position('bottom')
                                hide_axis(['right', 'top', 'left'], ax=ax)
                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)

                            xlim = limits.get(pars[i], (data.min(), data.max()))
                            ax.set_xlim(xlim)

                        if triangle == 'upper':
                            data_x = l[pars[i]]
                            data_y = l[pars[j]]
                            if i > j:

                                if i > j + 1:
                                        sharey = axes[j, i - 1]
                                else:
                                        sharey = None

                                ax = plt.subplot(len(pars), len(pars), k, sharey=sharey, sharex=sharex)
                                axes[j, i] = ax
                                if plotfunc is None:
                                        plt.plot(data_x, data_y, ',', **kwargs)
                                else:
                                        plotfunc(data_x, data_y, ax=ax, *args, **kwargs)
                                xlim = limits.get(pars[i], None)
                                ylim = limits.get(pars[j], None)
                                if xlim is not None:
                                    ax.set_xlim(xlim)
                                if ylim is not None:
                                    ax.set_ylim(ylim)

                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)
                                if devectorize is True:
                                    devectorize_axes(ax=ax)

                        if triangle == 'lower':
                            data_x = l[pars[i]]
                            data_y = l[pars[j]]
                            if i < j:

                                if i < j:
                                        sharey = axes[j, i - 1]
                                else:
                                        sharey = None

                                ax = plt.subplot(len(pars), len(pars), k, sharey=sharey, sharex=sharex)
                                axes[j, i] = ax
                                if plotfunc is None:
                                        plt.plot(data_x, data_y, ',', **kwargs)
                                else:
                                        plotfunc(data_x, data_y, ax=ax, *args, **kwargs)

                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)
                                xlim = limits.get(pars[i], None)
                                ylim = limits.get(pars[j], None)
                                if xlim is not None:
                                    ax.set_ylim(xlim)
                                if ylim is not None:
                                    ax.set_ylim(ylim)
                                if devectorize is True:
                                    devectorize_axes(ax=ax)

                            if i == 0:
                                ax.set_ylabel(lbls[j])
                                plt.setp(ax.get_yticklabels(), visible=True)

                            if j == len(pars) - 1:
                                ax.set_xlabel(lbls[i])
                                plt.setp(ax.get_xticklabels(), visible=True)

                        N = int(0.5 * fontmap[len(pars) - 1])
                        if N <= 4:
                            N = 5
                        setNmajors(N, N, ax=ax, prune='both')

                        k += 1
        setMargins(hspace=0.0, wspace=0.0)


def hinton(W, bg='grey', facecolors=('w', 'k')):
    """Draw a hinton diagram of the matrix W on the current pylab axis

    Hinton diagrams are a way of visualizing numerical values in a matrix/vector,
    popular in the neural networks and machine learning literature. The area
    occupied by a square is proportional to a value's magnitude, and the colour
    indicates its sign (positive/negative).

    Example usage:

        R = np.random.normal(0, 1, (2,1000))
        h, ex, ey = np.histogram2d(R[0], R[1], bins=15)
        hh = h - h.T
        hinton.hinton(hh)
    """
    M, N = W.shape
    square_x = np.array([-.5, .5, .5, -.5])
    square_y = np.array([-.5, -.5, .5, .5])

    ioff = False
    if plt.isinteractive():
        plt.ioff()
        ioff = True

    plt.fill([-.5, N - .5, N - .5, - .5], [-.5, -.5, M - .5, M - .5], bg)
    Wmax = np.abs(W).max()
    for m, Wrow in enumerate(W):
        for n, w in enumerate(Wrow):
            c = plt.signbit(w) and facecolors[1] or facecolors[0]
            plt.fill(square_x * w / Wmax + n, square_y * w / Wmax + m, c, edgecolor=c)

    plt.ylim(-0.5, M - 0.5)
    plt.xlim(-0.5, M - 0.5)

    if ioff is True:
        plt.ion()

    plt.draw_if_interactive()


def parallel_coordinates(d, labels=None, orientation='horizontal',
                         positions=None, ax=None, **kwargs):
    """ Plot parallel coordinates of a data set

    Each dimension is normalized and then plot either vertically or horizontally

    Parameters
    ----------
    d: ndarray, recarray or dict
        data to plot (one column or key per coordinate)

    labels: sequence
        sequence of string to use to define the label of each coordinate
        default p{:d}

    orientation: str
        'horizontal' of 'vertical' to set the plot orientation accordingly

    positions: sequence(float)
        position of each plane on the main axis. Default is equivalent to
        equidistant positioning.

    ax: plt.Axes instance
        axes to use for the figure, default plt.subplot(111)

    **kwargs: dict
        forwarded to :func:`plt.plot`
    """

    if labels is None:
        if hasattr(d, 'keys'):
            names = list(d.keys())
            data = np.array(d.values()).T
        elif hasattr(d, 'dtype'):
            if d.dtype.names is not None:
                names = d.dtype.names
            else:
                names = [ 'p{0:d}'.format(k) for k in range(len(d[0])) ]
        else:
                names = [ 'p{0:d}'.format(k) for k in range(len(d)) ]
        data = np.array(d).astype(float)
    else:
        names = labels
        data = np.array(d).astype(float)

    if len(labels) != len(data[0]):
        names = [ 'p{0:d}'.format(k) for k in range(len(data[0])) ]

    if positions is None:
        positions = np.arange(len(names))
    else:
        positions = np.array(positions)

    positions -= positions.min()
    dyn = np.ptp(positions)

    data = (data - data.mean(axis=0)) / data.ptp(axis=0)[None, :]
    order = np.argsort(positions)
    data = data[:, order]
    positions = positions[order]
    names = np.array(names)[order].tolist()

    if ax is None:
        ax = plt.subplot(111)

    if orientation.lower() == 'horizontal':
        ax.vlines(positions, -1, 1, color='0.8')
        ax.plot(positions, data.T, **kwargs)
        hide_axis(['left', 'right', 'top', 'bottom'], ax=ax)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(names)
        ax.set_xlim(positions.min() - 0.1 * dyn, positions.max() + 0.1 * dyn)
    else:
        ax.hlines(positions, -1, 1, color='0.8')
        ax.plot(data.T, positions, **kwargs)
        hide_axis(['left', 'right', 'top', 'bottom'], ax=ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_ylim(positions.min() - 0.1 * dyn, positions.max() + 0.1 * dyn)


def raw_string(seq):
    """ make a sequence of strings raw to avoid latex interpretation """

    def f(s):
        """ Filter latex """
        r = s.replace('\\', '\\\\').replace('_', '\_').replace('^', '\^')
        return r

    return [ f(k) for k in seq ]


def get_centers_from_bins(bins):
    """ return centers from bin sequence """
    return 0.5 * (bins[:-1] + bins[1:])


def nice_sigma_levels(im, sigs=[1, 2, 3]):
        """ Generates N sigma levels from a image map

        Parameters
        ----------

        H: ndarray
            values to find levels from

        sigs: sequence
            sigma levels to find

        Returns
        -------
        lvls: sequence
            Levels values corresponding to requested sigmas
        """
        V = 1.0 - np.exp(-0.5 * np.array(sigs) ** 2)
        Hflat = im.flatten()
        inds = np.argsort(Hflat)[::-1]
        Hflat = Hflat[inds]
        sm = np.cumsum(Hflat)
        sm /= sm[-1]

        for i, v0 in enumerate(V):
            try:
                V[i] = Hflat[sm <= v0][-1]
            except:
                V[i] = Hflat[0]
        return V


def nice_levels(H, N=5):
    """ Generates N sigma levels from a image map

    Parameters
    ----------

    H: ndarray
        values to find levels from

    N: int
        number of sigma levels to find

    Returns
    -------
    lvls: sequence
        Levels corresponding to 1..(N + 1) sigma levels
    """
    V = 1.0 - np.exp(-0.5 * np.arange(0.5, 0.5 * (N + 1 + 0.1), 0.5) ** 2)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    for i, v0 in enumerate(V):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    return V


def plot_density_map(x, y, xbins, ybins, Nlevels=4, cbar=True, weights=None):

    Z = np.histogram2d(x, y, bins=(xbins, ybins), weights=weights)[0].astype(float).T

    # central values
    lt = get_centers_from_bins(xbins)
    lm = get_centers_from_bins(ybins)
    cX, cY = np.meshgrid(lt, lm)
    X, Y = np.meshgrid(xbins, ybins)

    im = plt.pcolor(X, Y, Z, cmap=plt.cm.Blues)
    plt.contour(cX, cY, Z, levels=nice_levels(Z, Nlevels), cmap=plt.cm.Greys_r)

    if cbar:
        cb = plt.colorbar(im)
    else:
        cb = None
    plt.xlim(xbins[0], xbins[-1])
    plt.ylim(ybins[0], ybins[-1])

    try:
        plt.tight_layout()
    except Exception as e:
        print(e)
    return plt.gca(), cb


def triangle_plot(d, keys, bins=None, **kwargs):

    """
    Plot density maps all elements of gx against all elements of gy

    Parameters
    ----------
    d: dictionnary like structure
        data structure

    keys: sequence(str)
        keys from d to plot

    bins: sequence, optional
        bins to use per dimension
        default is adapted from the stddev

    labels: sequence(str)
        string to use as labels on the plots

    usekde: bool, optional, default: true
        if set use KDE to estimate densities, histograms otherwise

    tickrotation: float, optional, default: 0
        rotate the tick labels on the x-axis

    gaussian_ellipse: bool, optional, default: True
        if set, display the Gaussian error ellipse on top of each plot

    hpd: bool, optional, default: True
        if set display 1 and 3 sigma equivalent range on the 1d pdfs

    weights: ndarray
        weights to apply to each point

    returns
    -------
    axes: sequence
        all axes defined in the plot

    .. note::

        Other parameters are forwarded to :func:`plot_density_map`
        1d pdfs calls :func:`plot_1d_PDFs`
    """
    _drop = []
    ncols = len(keys)
    nlines = len(keys)
    shape = (nlines, ncols)
    axes = np.empty((nlines, ncols), dtype=object)
    lbls = kwargs.pop('labels', keys)
    usekde = kwargs.pop('usekde', True)
    ticksrotation = kwargs.pop('ticksrotation', 0)
    # gaussian_corner = kwargs.pop('gaussian_corner', False)
    gaussian_ellipse = kwargs.pop('gaussian_ellipse', False)
    hpd = kwargs.pop('hpd', True)
    weights = kwargs.pop('weights', None)
    max_n_ticks = kwargs.pop('max_n_ticks', 5)

    if bins is None:
        bins = []
        for k in keys:
            x = d[k]
            dx = 0.1 * np.std(x)
            bins.append(np.arange(x.min() - 2 * dx, x.max() + 2 * dx, dx))
    else:
        if not hasattr(bins, '__iter__'):
            bins = [bins] * len(keys)

    if len(bins) != len(keys):
        raise AttributeError('bins are not the same length as dimensions')

    for k in range(nlines * ncols):
        yk, xk = np.unravel_index(k, shape)
        kxk = keys[xk]
        kyk = keys[yk]
        if (xk >= 0) and (yk > xk):
            sharex = axes[xk, xk]
            sharey = axes[yk, 0]
        else:
            sharey = None
            sharex = None

        ax = plt.subplot(nlines, ncols, k + 1, sharey=sharey, sharex=sharex)
        if yk >= xk:
            axes[yk, xk] = ax
        # elif not gaussian_corner:
        else:
            _drop.append(ax)

        if (yk > xk):
            if usekde:
                kde = KDE_2d(d[kxk], d[kyk],
                             gridsize=(len(bins[xk]),len(bins[yk])),
                             adjust=0.5, weights=weights)
                kde.imshow()
                kde.contour(cmap=plt.cm.Greys_r)
            else:
                plot_density_map(d[kxk], d[kyk], bins[xk], bins[yk], cbar=False,
                                 weights=weights, **kwargs)

            if gaussian_ellipse & (weights is None):
                data = np.vstack([d[kxk], d[kyk]])
                mu = np.mean(data, axis=1)
                cov = np.cov(data - mu[:, None])
                error_ellipse(mu, cov, ax=ax, edgecolor="r", lw=2)
            if xk == 0:
                ax.set_ylabel(lbls[yk])
                [l.set_rotation(ticksrotation) for l in ax.get_yticklabels()]
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if yk == nlines - 1:
                ax.set_xlabel(lbls[xk])
                [l.set_rotation(ticksrotation) for l in ax.get_xticklabels()]
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.set_ylim(bins[yk][0], bins[yk][-1])
            ax.set_xlim(bins[xk][0], bins[xk][-1])
            ax.xaxis.set_label_coords(0.5, -0.6)
            ax.yaxis.set_label_coords(-0.6, 0.5)
        elif (yk == xk):
            if usekde:
                kde = KDE_1d(d[kxk], gridsize=len(bins[xk]), weights=weights)
                kde.plot(ec='k', alpha=0.8, lw=2, fc='w', facecolor='w')
                plt.vlines(kde.peak, ymin=0, ymax=kde.im.max(), color='r')
            else:
                n, _ = np.histogram(d[kxk], bins=bins[xk], weights=weights)
                xn = get_centers_from_bins(bins[xk])
                ax.fill_between(xn, [0] * len(xn), n.astype(float), edgecolor='k',
                                facecolor='w', alpha=0.8, lw=2)
                plt.vlines([xn[n.argmax()]], ymin=0, ymax=n.max(), color='r', lw=1, zorder=10)

                # ax.set_xlim(bins[xk][0], bins[xk][1])

            if hpd is True:
                ylim = ax.get_ylim()
                xn = percentile(d[kxk], (0.1, 15.7, 50, 84.3, 99.9), weights=weights)
                plt.vlines(xn, ymin=0, ymax=ylim[1], color='b', zorder=-10)
                plt.fill_between([xn[0], xn[2]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.05, zorder=-10)
                plt.fill_between([xn[1], xn[2]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.1, zorder=-10)
                plt.fill_between([xn[2], xn[-1]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.05, zorder=-10)
                plt.fill_between([xn[2], xn[-2]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.1, zorder=-10)
                ax.set_ylim(*ylim)
            plt.setp(ax.get_yticklabels(), visible=False)
            if yk == nlines - 1:
                ax.set_xlabel(lbls[xk])
                [l.set_rotation(ticksrotation) for l in ax.get_xticklabels()]
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklines(), visible=False)
            hide_axis(where=['top', 'left', 'right'], ax=ax)
            ax.set_xlim(bins[xk][0], bins[xk][-1])
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.xaxis.set_label_coords(0.5, -0.6)

    try:
       plt.tight_layout()
    except Exception as e:
        print(e)

    for ax in _drop:
        ax.set_visible(False)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    return [k for k in axes.ravel() if k is not None]


def plot_1d_PDFs(d, lbls, ticksrotation=45, hpd=True,
                 figout=None, usekde=True, bins=None, weights=None, ncols=4,
                 **kwargs):
    """
    Parameters
    ----------
    d: dictionnary like structure
        data structure

    keys: sequence(str)
        keys from d to plot

    bins: sequence, optional
        bins to use per dimension
        default is adapted from the stddev

    labels: sequence(str)
        string to use as labels on the plots

    usekde: bool, optional, default: true
        if set use KDE to estimate densities, histograms otherwise

    tickrotation: float, optional, default: 0
        rotate the tick labels on the x-axis

    hpd: bool, optional, default: True
        if set display 1 and 3 sigma equivalent range on the 1d pdfs

    weights: ndarray
        weights to apply on each point

    ncols: int
        number of columns

    returns
    -------
    axes: sequence
        all axes defined in the plot
    """
    lbls = list(lbls)

    if bins is None:
        bins = []
        for k in lbls:
            x = d[k]
            dx = 0.2 * np.std(x)
            bins.append(np.arange(x.min() - 2 * dx, x.max() + 2 * dx, dx))
    else:
        if not hasattr(bins, '__iter__'):
            bins = [bins] * len(lbls)

    if len(bins) != len(lbls):
        raise AttributeError('bins are not the same length as dimensions')

    xlabels = kwargs.pop('labels', lbls)

    ndim = len(lbls)
    nlines = max(1, ndim // ncols + int(ndim % (ndim // ncols) > 0))
    if nlines * ncols < len(lbls):
        nlines += 1
    plt.figure(figsize=( 4 * ncols, 4 * nlines ))

    axes = []
    for xk, kxk in enumerate(lbls):
        ax = plt.subplot(nlines, ncols, xk + 1)
        if usekde:
            kde = KDE_1d(d[kxk], gridsize=len(bins[xk]), weights=weights)
            kde.plot(ec='k', alpha=0.8, lw=2, fc='w', facecolor='w')
            plt.vlines(kde.peak, ymin=0, ymax=kde.im.max(), color='r')
        else:
            n, _ = np.histogram(d[kxk], bins=bins[xk], weights=weights)
            xn = get_centers_from_bins(bins[xk])
            ax.fill_between(xn, [0] * len(xn), n.astype(float), edgecolor='k',
                            facecolor='w', alpha=0.8, lw=2)
            plt.vlines([xn[n.argmax()]], ymin=0, ymax=n.max(), color='r', lw=1, zorder=10)

        if hpd is True:
            ylim = ax.get_ylim()
            xn = percentile(d[kxk], (0.1, 15.7, 50, 84.3, 99.9), weights=weights)
            plt.vlines(xn, ymin=0, ymax=ylim[1], color='b', zorder=-10)
            plt.fill_between([xn[0], xn[2]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.05, zorder=-10)
            plt.fill_between([xn[1], xn[2]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.1, zorder=-10)
            plt.fill_between([xn[2], xn[-1]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.05, zorder=-10)
            plt.fill_between([xn[2], xn[-2]], [0] * 2, [ylim[1]] * 2, color='b', alpha=0.1, zorder=-10)
            ax.set_ylim(*ylim)

        plt.setp(ax.get_yticklabels(), visible=False)
        [l.set_rotation(ticksrotation) for l in ax.get_xticklabels()]

        hide_axis(where=['top', 'left', 'right'], ax=ax)
        ax.set_xlabel(xlabels[xk])
        ax.set_ylim(0, ax.get_ylim()[1])
        axes.append(ax)

    try:
        tight_layout()
    except Exception as e:
        print(e)
    return axes

# =============================================================================
# Implementing THEMES in plt
# =============================================================================


class Theme(object):
    """This is an abstract base class for themes.

    In general, only complete themes should should subclass this class.


    Notes
    -----
    When subclassing there are really only two methods that need to be
    implemented.

    __init__: This should call super().__init__ which will define
    self._rcParams. Subclasses should customize self._rcParams after calling
    super().__init__. That will ensure that the rcParams are applied at
    the appropriate time.

    The other method is apply_theme(ax). This method takes an axes object that
    has been created during the plot process. The theme should modify the
    axes according.

    """

    _allowed_keys = plt.rcParams.keys()

    def __init__(self, *args, **kwargs):
        """
        Provide ggplot2 themeing capabilities.

        Parameters
        -----------
        kwargs**: theme_element
            kwargs are theme_elements based on http://docs.ggplot2.org/current/theme.html.
            Currently only a subset of the elements are implemented. In addition,
            Python does not allow using '.' in argument names, so we are using '_'
            instead.

            For example, ggplot2 axis.ticks.y will be  axis_ticks_y in Python ggplot.

        """
        self._rcParams = {}
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self.__setitem__(k, v)

    def __setitem__(self, k, v):
            if k in self._allowed_keys:
                self._rcParams[k] = v

    def keys(self, regexp=None, full_match=False):
        """
        Return the data column names or a subset of it

        Parameters
        ----------
        regexp: str
            pattern to filter the keys with

        full_match: bool
            if set, use :func:`re.fullmatch` instead of :func:`re.match`

        Try to apply the pattern at the start of the string, returning
        a match object, or None if no match was found.

        returns
        -------
        seq: sequence
            sequence of keys
        """
        keys = list(sorted(self._rcParams.keys()))

        if (regexp is None) or (regexp == '*'):
            return keys
        elif type(regexp) in basestring:
            if full_match is True:
                fn = re.fullmatch
            else:
                fn = re.match

            if regexp.count(',') > 0:
                _re = regexp.split(',')
            elif regexp.count(' ') > 0:
                _re = regexp.split()
            else:
                _re = [regexp]

            _keys = []
            for _rk in _re:
                _keys += [k for k in keys if (fn(_rk, k) is not None)]
            return _keys
        elif hasattr(regexp, '__iter__'):
            _keys = []
            for k in regexp:
                _keys += self.keys(k)
            return _keys
        else:
            raise ValueError('Unexpected type {0} for regexp'.format(type(regexp)))

    def apply_theme(self, ax):
        """apply_theme will be called with an axes object after plot has completed.

        Complete themes should implement this method if post plot themeing is
        required.
        """
        pass

    def get_rcParams(self):
        """Get an rcParams dict for this theme.

        Notes
        -----
        Subclasses should not need to override this method method as long as
        self._rcParams is constructed properly.

        rcParams are used during plotting. Sometimes the same theme can be
        achieved by setting rcParams before plotting or a post_plot_callback
        after plotting. The choice of how to implement it is is a matter of
        convenience in that case.

        There are certain things can only be themed after plotting. There
        may not be an rcParam to control the theme or the act of plotting
        may cause an entity to come into existence before it can be themed.

        """
        rcParams = deepcopy(self._rcParams)
        return rcParams

    def __add__(self, other):
        if isinstance(other, Theme):
            theme = deepcopy(self)
            theme.update(**other.get_rcParams())
            return theme
        else:
            raise TypeError()

    def post_callback(self, *args, **kwargs):
        pass

    def apply(self):
        self._rcstate = deepcopy(plt.rcParams)
        plt.rcParams.update(**self.get_rcParams())

    def restore(self):
        plt.rcParams.update(self._rcstate)

    def __enter__(self):
        self._rcstate = deepcopy(plt.rcParams)
        plt.rcParams.update(**self.get_rcParams())
        return self

    def __exit__(self, *args, **kwargs):
        self.post_callback()
        plt.rcParams.update(self._rcstate)

    def __call__(self, *args, **kwargs):
        return self.post_callback(*args, **kwargs)


class Theme_Seaborn(Theme):
    """
    Theme for seaborn.
    Copied from mwaskom's seaborn:
        https://github.com/mwaskom/seaborn/blob/master/seaborn/rcmod.py
    Parameters
    ----------
    style: whitegrid | darkgrid | nogrid | ticks
        Style of axis background.
    context: notebook | talk | paper | poster
        Intended context for resulting figures.
    gridweight: extra heavy | heavy | medium | light
        Width of the grid lines. None
    """

    def __init__(self, style="whitegrid", gridweight=None, context="notebook"):
        super(self.__class__, self).__init__()
        self.style = style
        self.gridweight = gridweight
        self.context = context
        self._set_theme_seaborn_rcparams(self._rcParams, self.style,
                                         self.gridweight, self.context)

    def _set_theme_seaborn_rcparams(self, rcParams, style, gridweight, context):
        """helper method to set the default rcParams and other theming relevant
        things
        """
        # select grid line width:
        gridweights = {'extra heavy': 1.5,
                       'heavy': 1.1,
                       'medium': 0.8,
                       'light': 0.5, }
        if gridweight is None:
            if context == "paper":
                glw = gridweights["medium"]
            else:
                glw = gridweights['extra heavy']
        elif np.isreal(gridweight):
            glw = gridweight
        else:
            glw = gridweights[gridweight]

        if style == "darkgrid":
            lw = .8 if context == "paper" else 1.5
            ax_params = {"axes.facecolor": "#EAEAF2",
                         "axes.edgecolor": "white",
                         "axes.linewidth": 0,
                         "axes.grid": True,
                         "axes.axisbelow": True,
                         "grid.color": "w",
                         "grid.linestyle": "-",
                         "grid.linewidth": glw}

        elif style == "whitegrid":
            lw = 1.0 if context == "paper" else 1.7
            ax_params = {"axes.facecolor": "white",
                         "axes.edgecolor": "#CCCCCC",
                         "axes.linewidth": lw,
                         "axes.grid": True,
                         "axes.axisbelow": True,
                         "grid.color": "#DDDDDD",
                         "grid.linestyle": "-",
                         "grid.linewidth": glw}

        elif style == "nogrid":
            ax_params = {"axes.grid": False,
                         "axes.facecolor": "white",
                         "axes.edgecolor": "black",
                         "axes.linewidth": 1}

        elif style == "ticks":
            ticksize = 3. if context == "paper" else 6.
            tickwidth = .5 if context == "paper" else 1
            ax_params = {"axes.grid": False,
                         "axes.facecolor": "white",
                         "axes.edgecolor": "black",
                         "axes.linewidth": 1,
                         "xtick.direction": "out",
                         "ytick.direction": "out",
                         "xtick.major.width": tickwidth,
                         "ytick.major.width": tickwidth,
                         "xtick.minor.width": tickwidth,
                         "xtick.minor.width": tickwidth,
                         "xtick.major.size": ticksize,
                         "xtick.minor.size": ticksize / 2,
                         "ytick.major.size": ticksize,
                         "ytick.minor.size": ticksize / 2}
        else:
            ax_params = {}

        rcParams.update(ax_params)

        # Determine the font sizes
        if context == "talk":
            font_params = {"axes.labelsize": 16,
                           "axes.titlesize": 19,
                           "xtick.labelsize": 14,
                           "ytick.labelsize": 14,
                           "legend.fontsize": 13,
                           }

        elif context == "notebook":
            font_params = {"axes.labelsize": 11,
                           "axes.titlesize": 12,
                           "xtick.labelsize": 10,
                           "ytick.labelsize": 10,
                           "legend.fontsize": 10,
                           }

        elif context == "poster":
            font_params = {"axes.labelsize": 18,
                           "axes.titlesize": 22,
                           "xtick.labelsize": 16,
                           "ytick.labelsize": 16,
                           "legend.fontsize": 16,
                           }

        elif context == "paper":
            font_params = {"axes.labelsize": 8,
                           "axes.titlesize": 12,
                           "xtick.labelsize": 8,
                           "ytick.labelsize": 8,
                           "legend.fontsize": 8,
                           }

        rcParams.update(font_params)

        # Set other parameters
        rcParams.update({
            "lines.linewidth": 1.1 if context == "paper" else 1.4,
            "patch.linewidth": .1 if context == "paper" else .3,
            "xtick.major.pad": 3.5 if context == "paper" else 7,
            "ytick.major.pad": 3.5 if context == "paper" else 7, })

        rcParams["timezone"] = "UTC"
        rcParams["patch.antialiased"] = "True"
        rcParams["font.family"] = "sans-serif"
        rcParams["font.size"] = "12.0"
        rcParams["font.serif"] = ["Times", "Palatino", "New Century Schoolbook",
                                  "Bookman", "Computer Modern Roman",
                                  "Times New Roman"]
        rcParams["font.sans-serif"] = ["Helvetica", "Avant Garde",
                                       "Computer Modern Sans serif", "Arial"]
        rcParams["axes.color_cycle"] = ["#333333", "348ABD", "7A68A6", "A60628",
                                        "467821", "CF4457", "188487", "E24A33"]
        rcParams["legend.fancybox"] = "True"
        rcParams["figure.figsize"] = "11, 8"
        rcParams["figure.facecolor"] = "1.0"
        rcParams["figure.edgecolor"] = "0.50"
        rcParams["figure.subplot.hspace"] = "0.5"

    def apply_theme(self, ax):
        """"Styles x,y axes to appear like ggplot2
        Must be called after all plot and axis manipulation operations have
        been carried out (needs to know final tick spacing)
        From: https://github.com/wrobstory/climatic/blob/master/climatic/stylers.py
        """
        # Remove axis border
        for child in ax.get_children():
            if isinstance(child, mpl.spines.Spine):
                child.set_alpha(0)

        # Restyle the tick lines
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(1.4)

        # Only show bottom left ticks
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # Set minor grid lines
        ax.grid(True, 'minor', color='#F2F2F2', linestyle='-', linewidth=0.7)

        if not isinstance(ax.xaxis.get_major_locator(), mpl.ticker.LogLocator):
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        if not isinstance(ax.yaxis.get_major_locator(), mpl.ticker.LogLocator):
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))


class Theme_538(Theme):
    """
    Theme for 538.
    http://dataorigami.net/blogs/napkin-folding/17543615-replicating-538s-plot-styles-in-matplotlib
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self._rcParams["lines.linewidth"] = "2.0"
        self._rcParams["patch.linewidth"] = "0.5"
        self._rcParams["legend.fancybox"] = "True"
        self._rcParams["axes.color_cycle"] = [ "#30a2da", "#fc4f30", "#e5ae38",
                                               "#6d904f", "#8b8b8b"]
        self._rcParams["axes.facecolor"] = "#f0f0f0"
        self._rcParams["axes.labelsize"] = "large"
        self._rcParams["axes.axisbelow"] = "True"
        self._rcParams["axes.grid"] = "True"
        self._rcParams["patch.edgecolor"] = "#f0f0f0"
        self._rcParams["axes.titlesize"] = "x-large"
        self._rcParams["svg.embed_char_paths"] = "path"
        self._rcParams["figure.facecolor"] = "#f0f0f0"
        self._rcParams["grid.linestyle"] = "-"
        self._rcParams["grid.linewidth"] = "1.0"
        self._rcParams["grid.color"] = "#cbcbcb"
        self._rcParams["axes.edgecolor"] = "#f0f0f0"
        self._rcParams["xtick.major.size"] = "0"
        self._rcParams["xtick.minor.size"] = "0"
        self._rcParams["ytick.major.size"] = "0"
        self._rcParams["ytick.minor.size"] = "0"
        self._rcParams["axes.linewidth"] = "3.0"
        self._rcParams["font.size"] = "14.0"
        self._rcParams["lines.linewidth"] = "4"
        self._rcParams["lines.solid_capstyle"] = "butt"
        self._rcParams["savefig.edgecolor"] = "#f0f0f0"
        self._rcParams["savefig.facecolor"] = "#f0f0f0"
        self._rcParams["figure.subplot.left"]   = "0.08"
        self._rcParams["figure.subplot.right"]  = "0.95"
        self._rcParams["figure.subplot.bottom"] = "0.07"

    def apply_theme(self, ax):
        '''Styles x,y axes to appear like ggplot2
        Must be called after all plot and axis manipulation operations have
        been carried out (needs to know final tick spacing)
        From: https://github.com/wrobstory/climatic/blob/master/climatic/stylers.py
        '''
        # Remove axis border
        for child in ax.get_children():
            if isinstance(child, mpl.spines.Spine):
                child.set_alpha(0)

        # Restyle the tick lines
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(1.4)

        # Only show bottom left ticks
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # Set minor grid lines
        ax.grid(True, 'minor', color='#F2F2F2', linestyle='-', linewidth=0.7)

        if not isinstance(ax.xaxis.get_major_locator(), mpl.ticker.LogLocator):
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        if not isinstance(ax.yaxis.get_major_locator(), mpl.ticker.LogLocator):
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))


class Theme_Gray(Theme):
    """
    Standard theme for ggplot. Gray background w/ white gridlines.
    Copied from the the ggplot2 codebase:
        https://github.com/hadley/ggplot2/blob/master/R/theme-defaults.r
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self._rcParams["timezone"] = "UTC"
        self._rcParams["lines.linewidth"] = "1.0"
        self._rcParams["lines.antialiased"] = "True"
        self._rcParams["patch.linewidth"] = "0.5"
        self._rcParams["patch.facecolor"] = "348ABD"
        self._rcParams["patch.edgecolor"] = "#E5E5E5"
        self._rcParams["patch.antialiased"] = "True"
        self._rcParams["font.family"] = "sans-serif"
        self._rcParams["font.size"] = "12.0"
        self._rcParams["font.serif"] = ["Times", "Palatino",
                                        "New Century Schoolbook",
                                        "Bookman", "Computer Modern Roman",
                                        "Times New Roman"]
        self._rcParams["font.sans-serif"] = ["Helvetica", "Avant Garde",
                                             "Computer Modern Sans serif",
                                             "Arial"]
        self._rcParams["axes.facecolor"] = "#E5E5E5"
        self._rcParams["axes.edgecolor"] = "bcbcbc"
        self._rcParams["axes.linewidth"] = "1"
        self._rcParams["axes.grid"] = "True"
        self._rcParams["axes.titlesize"] = "x-large"
        self._rcParams["axes.labelsize"] = "large"
        self._rcParams["axes.labelcolor"] = "black"
        self._rcParams["axes.axisbelow"] = "True"
        self._rcParams["axes.color_cycle"] = ["#333333", "348ABD", "7A68A6",
                                              "A60628",
                                              "467821", "CF4457", "188487",
                                              "E24A33"]
        self._rcParams["grid.color"] = "white"
        self._rcParams["grid.linewidth"] = "1.4"
        self._rcParams["grid.linestyle"] = "solid"
        self._rcParams["xtick.major.size"] = "0"
        self._rcParams["xtick.minor.size"] = "0"
        self._rcParams["xtick.major.pad"] = "6"
        self._rcParams["xtick.minor.pad"] = "6"
        self._rcParams["xtick.color"] = "#444444"
        self._rcParams["xtick.direction"] = "out"  # pointing out of axis
        self._rcParams["ytick.major.size"] = "0"
        self._rcParams["ytick.minor.size"] = "0"
        self._rcParams["ytick.major.pad"] = "6"
        self._rcParams["ytick.minor.pad"] = "6"
        self._rcParams["ytick.color"] = "#444444"
        self._rcParams["ytick.direction"] = "out"  # pointing out of axis
        self._rcParams["legend.fancybox"] = "True"
        self._rcParams["figure.figsize"] = "11, 8"
        self._rcParams["figure.facecolor"] = "1.0"
        self._rcParams["figure.edgecolor"] = "0.50"
        self._rcParams["figure.subplot.hspace"] = "0.5"

    def apply_theme(self, ax):
        '''Styles x,y axes to appear like ggplot2
        Must be called after all plot and axis manipulation operations have
        been carried out (needs to know final tick spacing)
        From: https://github.com/wrobstory/climatic/blob/master/climatic/stylers.py
        '''
        # Remove axis border
        for child in ax.get_children():
            if isinstance(child, mpl.spines.Spine):
                child.set_alpha(0)

        # Restyle the tick lines
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(1.4)

        # Only show bottom left ticks
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # Set minor grid lines
        ax.grid(True, 'minor', color='#F2F2F2', linestyle='-', linewidth=0.7)

        if not isinstance(ax.xaxis.get_major_locator(), mpl.ticker.LogLocator):
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        if not isinstance(ax.yaxis.get_major_locator(), mpl.ticker.LogLocator):
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))


class Theme_Matplotlib(Theme):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.update(**plt.rcParamsDefault)
        plt.ion()


class Theme_Ezrc(Theme):
    def __init__(self, fontSize=16., lineWidth=1., labelSize=None,
                 tickmajorsize=10, tickminorsize=5, figsize=(8, 6)):

        if labelSize is None:
            labelSize = fontSize + 2
        rcParams = {}
        rcParams['figure.figsize'] = figsize
        rcParams['lines.linewidth'] = lineWidth
        rcParams['grid.linewidth'] = lineWidth
        rcParams['font.sans-serif'] = ['Helvetica']
        rcParams['font.serif'] = ['Helvetica']
        rcParams['font.family'] = ['Times New Roman']
        rcParams['font.size'] = fontSize
        rcParams['font.family'] = 'serif'
        rcParams['font.weight'] = 'bold'
        rcParams['axes.linewidth'] = lineWidth
        rcParams['axes.labelsize'] = labelSize
        rcParams['legend.borderpad'] = 0.1
        rcParams['legend.markerscale'] = 1.
        rcParams['legend.fancybox'] = False
        rcParams['text.usetex'] = True
        rcParams['image.aspect'] = 'auto'
        rcParams['ps.useafm'] = True
        rcParams['ps.fonttype'] = 3
        rcParams['xtick.major.size'] = tickmajorsize
        rcParams['xtick.minor.size'] = tickminorsize
        rcParams['ytick.major.size'] = tickmajorsize
        rcParams['ytick.minor.size'] = tickminorsize
        rcParams['text.latex.preamble'] = ["\\usepackage{amsmath}"]
        super(self.__class__, self).__init__(**rcParams)
        plt.ion()


class Theme_Latex(Theme):
    def __init__(self, fontSize=None, labelSize=None):

        rcParams = {}
        if fontSize is not None:
            if labelSize is None:
                labelSize = fontSize

            rcParams['font.sans-serif'] = ['Helvetica']
            rcParams['font.serif'] = ['Helvetica']
            rcParams['font.family'] = ['Times New Roman']
            rcParams['font.size'] = fontSize
            rcParams["axes.labelsize"] = labelSize
            rcParams["axes.titlesize"] = labelSize
            rcParams["xtick.labelsize"] = labelSize
            rcParams["ytick.labelsize"] = labelSize
            rcParams["legend.fontsize"] = fontSize
            rcParams['font.family'] = 'serif'
            rcParams['font.weight'] = 'bold'
            rcParams['axes.labelsize'] = labelSize
            rcParams['text.usetex'] = True
            rcParams['ps.useafm'] = True
            rcParams['ps.fonttype'] = 3
            rcParams['text.latex.preamble'] = ["\\usepackage{amsmath}"]

        super(self.__class__, self).__init__(**rcParams)
        plt.ion()


class Theme_TightLayout(Theme):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(self.__class__, self).__init__()

    def post_callback(self, *args, **kwargs):
        self.kwargs.update(kwargs)
        tight_layout(**self.kwargs)


def declare_parula():
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import cm

    cm_data = [
        [0.2081, 0.1663, 0.5292],
        [0.2116238095, 0.1897809524, 0.5776761905],
        [0.212252381, 0.2137714286, 0.6269714286],
        [0.2081, 0.2386, 0.6770857143],
        [0.1959047619, 0.2644571429, 0.7279],
        [0.1707285714, 0.2919380952, 0.779247619],
        [0.1252714286, 0.3242428571, 0.8302714286],
        [0.0591333333, 0.3598333333, 0.8683333333],
        [0.0116952381, 0.3875095238, 0.8819571429],
        [0.0059571429, 0.4086142857, 0.8828428571],
        [0.0165142857, 0.4266, 0.8786333333],
        [0.032852381, 0.4430428571, 0.8719571429],
        [0.0498142857, 0.4585714286, 0.8640571429],
        [0.0629333333, 0.4736904762, 0.8554380952],
        [0.0722666667, 0.4886666667, 0.8467],
        [0.0779428571, 0.5039857143, 0.8383714286],
        [0.079347619, 0.5200238095, 0.8311809524],
        [0.0749428571, 0.5375428571, 0.8262714286],
        [0.0640571429, 0.5569857143, 0.8239571429],
        [0.0487714286, 0.5772238095, 0.8228285714],
        [0.0343428571, 0.5965809524, 0.819852381],
        [0.0265, 0.6137, 0.8135],
        [0.0238904762, 0.6286619048, 0.8037619048],
        [0.0230904762, 0.6417857143, 0.7912666667],
        [0.0227714286, 0.6534857143, 0.7767571429],
        [0.0266619048, 0.6641952381, 0.7607190476],
        [0.0383714286, 0.6742714286, 0.743552381],
        [0.0589714286, 0.6837571429, 0.7253857143],
        [0.0843, 0.6928333333, 0.7061666667],
        [0.1132952381, 0.7015, 0.6858571429],
        [0.1452714286, 0.7097571429, 0.6646285714],
        [0.1801333333, 0.7176571429, 0.6424333333],
        [0.2178285714, 0.7250428571, 0.6192619048],
        [0.2586428571, 0.7317142857, 0.5954285714],
        [0.3021714286, 0.7376047619, 0.5711857143],
        [0.3481666667, 0.7424333333, 0.5472666667],
        [0.3952571429, 0.7459, 0.5244428571],
        [0.4420095238, 0.7480809524, 0.5033142857],
        [0.4871238095, 0.7490619048, 0.4839761905],
        [0.5300285714, 0.7491142857, 0.4661142857],
        [0.5708571429, 0.7485190476, 0.4493904762],
        [0.609852381, 0.7473142857, 0.4336857143],
        [0.6473, 0.7456, 0.4188],
        [0.6834190476, 0.7434761905, 0.4044333333],
        [0.7184095238, 0.7411333333, 0.3904761905],
        [0.7524857143, 0.7384, 0.3768142857],
        [0.7858428571, 0.7355666667, 0.3632714286],
        [0.8185047619, 0.7327333333, 0.3497904762],
        [0.8506571429, 0.7299, 0.3360285714],
        [0.8824333333, 0.7274333333, 0.3217],
        [0.9139333333, 0.7257857143, 0.3062761905],
        [0.9449571429, 0.7261142857, 0.2886428571],
        [0.9738952381, 0.7313952381, 0.266647619],
        [0.9937714286, 0.7454571429, 0.240347619],
        [0.9990428571, 0.7653142857, 0.2164142857],
        [0.9955333333, 0.7860571429, 0.196652381],
        [0.988, 0.8066, 0.1793666667],
        [0.9788571429, 0.8271428571, 0.1633142857],
        [0.9697, 0.8481380952, 0.147452381],
        [0.9625857143, 0.8705142857, 0.1309],
        [0.9588714286, 0.8949, 0.1132428571],
        [0.9598238095, 0.9218333333, 0.0948380952],
        [0.9661, 0.9514428571, 0.0755333333],
        [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    cm.register_cmap('parula', cmap=parula_map)
    cm.__dict__['parula'] = cm.get_cmap('parula')
    parula_r_map = LinearSegmentedColormap.from_list('parula_r', cm_data[::-1])
    cm.register_cmap('parula_r', cmap=parula_r_map)
    cm.__dict__['parula_r'] = cm.get_cmap('parula_r')

    parula_map = LinearSegmentedColormap.from_list('parulaW', [[1., 1., 1.]] + cm_data + [[1., 1., 1.]])
    cm.register_cmap('parulaW', cmap=parula_map)
    cm.__dict__['parulaW'] = cm.get_cmap('parulaW')
    parula_r_map = LinearSegmentedColormap.from_list('parulaW_r', ([[1., 1., 1.]] + cm_data + [[1., 1., 1.]])[::-1])
    cm.register_cmap('parulaW_r', cmap=parula_r_map)
    cm.__dict__['parulaW_r'] = cm.get_cmap('parulaW_r')
