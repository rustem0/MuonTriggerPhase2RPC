from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import sys

#----------------------------------------------------------------------------------------------
def waitForClick(figname=None, saveAll=True, figure=None):

    if figure:
        figure.show()

    safeFig = False

    while True:
        try:
            result = plt.waitforbuttonpress()

            if result == True:
                saveFig = True
            
            break
        except:
            sys.exit(0)

#----------------------------------------------------------------------------------------------
def makeKDE2d(x, y, bandwidth, xbins=100j, ybins=100j, maskzero=True, kernel='tophat', **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kwargs)

    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))

    zz = np.reshape(z, xx.shape)

    if maskzero:
        zz[zz == 0] = np.nan

    return xx, yy, zz

#----------------------------------------------------------------------------------------------
def plotKDE2d(xx, yy, zz, cname, xtitle, ytitle, labelSize=20, labelPad=-1):

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 10))
    plt.subplots_adjust(bottom=0.08, left=0.10, top=0.98, right=0.99)

    cmap = plt.get_cmap(cname)

    im = ax.pcolormesh(xx, yy, zz, cmap=cmap)

    cbar = fig.colorbar(im, ax=ax, fraction=0.15, pad=0.005)
    cbar.ax.tick_params(labelsize=labelSize) 

    ax.set_xlabel(xtitle, fontsize=labelSize, labelpad=labelPad)
    ax.set_ylabel(ytitle, fontsize=labelSize, labelpad=labelPad)

    ax.tick_params(axis='x', labelsize=labelSize)
    ax.tick_params(axis='y', labelsize=labelSize)

#----------------------------------------------------------------------------------------------
def getcmaps():

    cmaps = {}
    
    cmaps['Perceptually Uniform Sequential'] = [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis']

    cmaps['Sequential'] = [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    return cmaps

#----------------------------------------------------------------------------------------------
def main():
    m1 = np.random.normal(size=20000)
    m2 = np.random.normal(scale=0.5, size=20000)

    x, y = m1 + m2, m1 - m2

    xx, yy, zz = makeKDE2d(x, y, 0.5)
                
    cmaps = getcmaps()

    for clist in cmaps.values():
        for cname in clist:
            plotKDE2d(xx, yy, zz, cname, r'$x_{1}$', r'$y_{1}$', )
            waitForClick()

#----------------------------------------------------------------------------------------------
main()
