import os, sys, pathlib, time, re, glob, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore");

######################################################################################################
SNUMBER = pow(10, -124);
def uniform(lowX, highX, x):
    xArr = np.array(x)
    if (xArr < lowX) or (highX < xArr): return SNUMBER
    return np.ones_like(xArr)/(highX-lowX)
def gaussian(mu, sig, x):
    X = np.array(x)
    val = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))*(1.0/(sig*np.sqrt(2.0*np.pi)))
    val[val < SNUMBER] = SNUMBER
    return val

def priorFull(a, b):      return lambda x : uniform(a, b, x)
def likelihoodFull(b, x): return lambda a : gaussian(a, b, x)
#######################################################################################################
#plotMargin: left, bottom, right, top
DEFAULT_SCALEYX=((1.0-0.1-0.11)*7.0)/((1.0-0.13-0.08)*9.0)
def getSizeMargin(gridSpec, subplotSize=[9.0, 7.0], plotMargin=[0.13, 0.1, 0.08, 0.11]):
    figSize = (gridSpec[1]*subplotSize[0], gridSpec[0]*subplotSize[1])
    marginRatio = [plotMargin[0]/gridSpec[1], plotMargin[1]/gridSpec[0],\
                   1.0 - plotMargin[2]/gridSpec[1], 1.0 - plotMargin[3]/gridSpec[0],\
                   (plotMargin[0] + plotMargin[2])/(1.0 - plotMargin[0] - plotMargin[2]),\
                   (plotMargin[1] + plotMargin[3])/(1.0 - plotMargin[1] - plotMargin[3])]
    return figSize, marginRatio
    
######################################################################################################
def main():
    verbosity = 1
    binN = 200
    rangeX = [-1.0, 11.0]
    rangeY = [rangeX[0], rangeX[0] + (1195.0/1610)*(rangeX[1] - rangeX[0])]

    np.random.seed(2)
    class1Mu  = [7.0, 2.0]
    class1Cov = [[1.5, -0.3], [-0.9, 0.9]]
    class1N   = 10000
    class2Mu  = [4.0, 4.0]
    class2Cov = [[0.8, 1.0], [1.5, 1.1]]
    class2N   = 20000

#training data
    class1Data = np.transpose(np.random.multivariate_normal(class1Mu, class1Cov, class1N))
    class2Data = np.transpose(np.random.multivariate_normal(class2Mu, class2Cov, class2N))
#plots
    gridSpec = [1, 1]
    figSize, marginRatio = getSizeMargin(gridSpec, subplotSize=[18.0, 14.0])
    fig = plt.figure(figsize=figSize); fig.subplots_adjust(*marginRatio)
    gs = gridspec.GridSpec(*gridSpec)
    matplotlib.rc('xtick', labelsize=24)
    matplotlib.rc('ytick', labelsize=24)
    ax = []
    for axIdx in range(gridSpec[0]*gridSpec[1]):
        ax.append(fig.add_subplot(gs[axIdx]));
        ax[-1].ticklabel_format(style='sci', scilimits=(-2, 2))
        ax[-1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
 
    #plot 0
    ax[0].scatter(*class1Data, marker="o", s=40.0, color="orange", alpha=0.2,edgecolors="none")
    ax[0].scatter(*class2Data, marker="o", s=40.0, color="blue",   alpha=0.2,edgecolors="none")
    ax[0].set_title("Generated Training Data", fontsize=40, y=1.03)
    ax[0].set_xlabel("feature x", fontsize=32)
    ax[0].set_ylabel("feature y", fontsize=32)
    ax[0].set_xlim(*rangeX)
    ax[0].set_ylim(*rangeY)
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/naiveBayes2gaus.png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig)
    if verbosity >= 1: print("Creating the following files:\n ", filenameFig)

######################################################################################################
if __name__ == "__main__": main()




