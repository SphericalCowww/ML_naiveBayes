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
def getClosestIdx(arr, val): #too lazy to use the bisection method
    minDiff = 1.0E12
    minIdx = None
    for arrIdx, arrVal in enumerate(arr): 
        if abs(val - arrVal) < minDiff:
            minDiff = abs(val - arrVal)
            minIdx = arrIdx
    return minIdx
     
######################################################################################################
def main():
    verbosity = 1
    binN = 200
    rangeX = [-5.0, 15.0]
    rangeY = [rangeX[0], rangeX[0] + (1195.0/1610)*(rangeX[1] - rangeX[0])]

    np.random.seed(3)
    class1Mu  = [7.0, 2.0]
    class1Cov = [[1.5, -0.3], [-0.9, 0.9]]
    class1N   = 10000
    class2Mu  = [4.0, 4.0]
    class2Cov = [[0.8, 1.0], [1.5, 1.1]]
    class2N   = 20000

    testN = 100
#training data
    class1Data = np.transpose(np.random.multivariate_normal(class1Mu, class1Cov, class1N))
    class2Data = np.transpose(np.random.multivariate_normal(class2Mu, class2Cov, class2N))
    testData = [np.random.uniform(*rangeX, size=testN), np.random.uniform(*rangeY, size=testN)]

    projX = np.linspace(*rangeX, binN+1)[:-1]
    projY = np.linspace(*rangeY, binN+1)[:-1]
    class1XHist = np.histogram(class1Data[0], bins=binN, range=rangeX)[0] #bin2left
    class1YHist = np.histogram(class1Data[1], bins=binN, range=rangeY)[0]
    class2XHist = np.histogram(class2Data[0], bins=binN, range=rangeX)[0]
    class2YHist = np.histogram(class2Data[1], bins=binN, range=rangeY)[0]

    class1XHist = 1.0*class1XHist + 0.001
    class1YHist = 1.0*class1YHist + 0.001
    class2XHist = 1.0*class2XHist + 0.001
    class2YHist = 1.0*class2YHist + 0.001
    class1XHist = class1XHist/(np.sum(class1XHist)*(projX[1]-projX[0]))
    class1YHist = class1YHist/(np.sum(class1YHist)*(projY[1]-projY[0]))
    class2XHist = class2XHist/(np.sum(class2XHist)*(projX[1]-projX[0]))
    class2YHist = class2YHist/(np.sum(class2YHist)*(projY[1]-projY[0]))
    class1XHistMin = np.min(class1XHist)
    class1YHistMin = np.min(class1YHist)
    class2XHistMin = np.min(class2XHist)
    class2YHistMin = np.min(class2YHist)
#classification: instead of the discrete case, can also use fitted gaussians to evaluate the scores
    testAmbiguous = [[rangeX[0]-1.0, rangeY[0]-1.0]]
    testClass1    = [[rangeX[0]-1.0, rangeY[0]-1.0]]
    testClass2    = [[rangeX[0]-1.0, rangeY[0]-1.0]]
    if verbosity >= 1: print("Start classification...")
    for dataPoint in np.transpose(testData):
        projXidx = getClosestIdx(projX, dataPoint[0])
        projYidx = getClosestIdx(projY, dataPoint[1])
        class1Prior = len(class1Data)/(len(class1Data) + len(class2Data))
        class1Score = np.log(class1Prior)+np.log(class1XHist[projXidx])+np.log(class1YHist[projYidx])
        class2Prior = len(class2Data)/(len(class1Data) + len(class2Data))
        class2Score = np.log(class2Prior)+np.log(class2XHist[projXidx])+np.log(class2YHist[projYidx])
        ambiCond = (abs(class1Score - class2Score) < 1.0)
        ambiCond = ambiCond or ((class1XHist[projXidx] < 2.0*class1XHistMin) and
                                (class1YHist[projYidx] < 2.0*class1YHistMin) and 
                                (class2XHist[projXidx] < 2.0*class2XHistMin) and
                                (class2YHist[projYidx] < 2.0*class2YHistMin))
        if ambiCond:                    testAmbiguous.append(dataPoint)
        elif class1Score > class2Score: testClass1   .append(dataPoint)
        else:                           testClass2   .append(dataPoint)
        if verbosity >= 2:
            print("  dataPoint="+str(dataPoint)+" => score12diff = "+\
                  str([class1Score, class2Score, abs(class1Score - class2Score)]))
    testAmbiguous = np.transpose(testAmbiguous)
    testClass1    = np.transpose(testClass1)
    testClass2    = np.transpose(testClass2)
#plots
    gridSpec = [2, 2]
    figSize, marginRatio = getSizeMargin(gridSpec, subplotSize=[18.0, 14.0])
    fig = plt.figure(figsize=figSize); fig.subplots_adjust(*marginRatio)
    gs = gridspec.GridSpec(*gridSpec)
    matplotlib.rc('xtick', labelsize=32)
    matplotlib.rc('ytick', labelsize=32)
    ax = []
    for axIdx in range(gridSpec[0]*gridSpec[1]):
        ax.append(fig.add_subplot(gs[axIdx]));
        ax[-1].ticklabel_format(style='sci', scilimits=(-2, 2))
        ax[-1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].tick_params(which="major", width=3, length=10)
        ax[-1].tick_params(which="minor", width=2, length=6)
 
    ax[0].scatter(*class1Data, marker="o", s=40.0, color="orange", alpha=0.2, edgecolors="none")
    ax[0].scatter(*class2Data, marker="o", s=40.0, color="green",  alpha=0.2, edgecolors="none")
    ax[0].scatter(*testData,   marker="*", s=600.0, color="black", edgecolors="none")
    ax[0].set_title("Generated Training Data", fontsize=48, y=1.03)
    ax[0].set_xlabel("feature x", fontsize=40)
    ax[0].set_ylabel("feature y", fontsize=40)
    ax[0].set_xlim(*rangeX)
    ax[0].set_ylim(*rangeY)

    ax[1].hist(class1Data[1], bins=binN, range=rangeY, histtype='step', color='orange', linewidth=4,\
               density=True, orientation='horizontal')
    ax[1].hist(class2Data[1], bins=binN, range=rangeY, histtype='step', color='green', linewidth=4,\
               density=True, orientation='horizontal')
    ax[1].set_title("Feature Y Projection", fontsize=48, y=1.03)
    ax[1].set_xlabel('count', fontsize=40)
    ax[1].set_ylabel('feature y', fontsize=40)
    ax[1].set_xlim(left=0.0)
    ax[1].set_ylim(*rangeY)

    ax[2].plot(projX, class1XHist, drawstyle="steps-post", color='orange', linewidth=4)
    ax[2].plot(projX, class2XHist, drawstyle="steps-post", color='green',  linewidth=4)
    ax[2].set_title("Feature X Projection", fontsize=48, y=1.03)
    ax[2].set_xlabel('feature x', fontsize=40)
    ax[2].set_ylabel('count', fontsize=40)
    ax[2].set_xlim(*rangeX)
    ax[2].invert_yaxis()
    ax[2].set_ylim(top=0)

    plot0 = ax[3].scatter(*testAmbiguous, marker="*", s=600.0, color="grey",   edgecolors="none")
    plot1 = ax[3].scatter(*testClass1,    marker="*", s=600.0, color="orange", edgecolors="none")
    plot2 = ax[3].scatter(*testClass2,    marker="*", s=600.0, color="green",  edgecolors="none")
    ax[3].set_title("Test Data Classification", fontsize=48, y=1.03)
    ax[3].set_xlabel("feature x", fontsize=40) 
    ax[3].set_ylabel("feature y", fontsize=40)
    ax[3].set_xlim(*rangeX)
    ax[3].set_ylim(*rangeY)
    legObj = ax[3].legend([plot0, plot1, plot2], ["ambiguous", "class as orange", "class as green"],\
                          loc="upper right", fontsize=24)
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/naiveBayes2gaus.png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig)
    if verbosity >= 1: print("Creating the following files:\n ", filenameFig)

######################################################################################################
if __name__ == "__main__": main()




