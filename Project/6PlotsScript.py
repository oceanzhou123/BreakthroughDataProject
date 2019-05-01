import blimpy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import h5py
import argparse
import os
"""Creating our argparser which will take in command line arguments. There are 2 mandatory arguments: the filepaths of the cadence you wish to analyze
and the file you want to save the 6 plots of the observation as (if you don't want to save then type None). The other 4 optional arguments are if you
are extracting the plots for a certain time and/or frequency range."""

parser = argparse.ArgumentParser(description='Enter a list of filepaths and format of saving the 6 waterfall plots')
parser.add_argument(
    'filepaths',
    help = 'The filepath of the zipfile representing observation cadence',
    nargs = '+',
)
parser.add_argument(
    'saveAs',
    help = 'The type of file you wish to save it as',
    type = str,
)
parser.add_argument(
    '--freqstart',
    help = 'The start frequency for the section of data you wish to extract',
    type = float,
)
parser.add_argument(
    '--freqend',
    help = 'The end frequency for the section of data you wish to extract',
    type = float,
)
parser.add_argument(
    '--tstart',
    help = 'The start time for the section of data you wish to extract',
    type = float,
)
parser.add_argument(
    '--tend',
    help = 'The end time for the section of data you wish to extract',
    type = float,
)

args = parser.parse_args()
def h5_header_wrapper(filename):
    """Gets header of a .h5 file."""
    h = h5py.File(filename)
    header = dict(h['data'].attrs.items())
    return header

def waterfallfunc(filepath):
    """
    Reads the given filepath into waterfall filterbank file
    """
    return(blimpy.Waterfall(filepath))

filepaths = args.filepaths
waterfalls = [waterfallfunc(filepath) for filepath in filepaths]
hd = [h5_header_wrapper(filepath) for filepath in filepaths]
titles = [str(hd[0]['source_name']), str(hd[1]['source_name']), str(hd[2]['source_name']), str(hd[3]['source_name']), str(hd[4]['source_name']), str(hd[5]['source_name'])]
titles = [titles[0][2:-1].upper(), titles[1][2:-1].upper(), titles[2][2:-1].upper(), titles[3][2:-1].upper(), titles[4][2:-1].upper(), titles[5][2:-1].upper()]

def waterfallTime(waterfall):
    """
    Returns the timestamps of the waterfall file's data
    """
    return(waterfall.populate_timestamps())

def waterfallFreqs(waterfall):
    """
    Returns the frequencies of the waterfall file's data
    """
    return(waterfall.populate_freqs())

def waterfallData(waterfall):
    """
    Extracts the actual data of the waterfall file
    """
    return(waterfall.data[:, 0, :])

def waterfallSize(waterfall):
    """
    Extracts the size of the data of the waterfall file
    """
    return(waterfall.data[:, 0, :].shape)

times = [waterfallTime(waterfall) for waterfall in waterfalls]
freqs = [waterfallFreqs(waterfall) for waterfall in waterfalls]
data = [waterfallData(waterfall) for waterfall in waterfalls]
size = [waterfallSize(waterfall) for waterfall in waterfalls]

def elapsedTime(times):
    """
    Converting timestamps from MJD time into seconds elapsed since first observation
    """
    timesElapsedDays = times - times[0]
    timesElapsedSeconds = timesElapsedDays * 86400
    return(timesElapsedSeconds)

telapsed = [elapsedTime(time) for time in times]

def locateTime(obsNumber, vertIndex):
    """
    Extracting the time of each datapoint, given the vertical index of the
    datapoint and the obsNumber. The obsNumber is the observation number of the waterfall
    file, or in other words, denotes the place of the observation in the cadence. obsNumber
    is 1-6.
    """
    if obsNumber == 1:
        time = telapsed[0][vertIndex]
    elif obsNumber == 2:
        time = telapsed[1][vertIndex]
    elif obsNumber == 3:
        time = telapsed[2][vertIndex]
    elif obsNumber == 4:
        time = telapsed[3][vertIndex]
    elif obsNumber == 5:
        time = telapsed[4][vertIndex]
    elif obsNumber == 6:
        time = telapsed[5][vertIndex]
    return(time)

def locateFreq(obsNumber, horizIndex):
    """
    Extracting the frequency of each datapoint, given the vertical index of the
    datapoint and the obsNumber. The obsNumber is the observation number of the waterfall
    file, or in other words, denotes the place of the observation in the cadence. obsNumber
    is 1-6.
    """
    if obsNumber == 1:
        freq = freqs[0][horizIndex]
    elif obsNumber == 2:
        freq = freqs[1][horizIndex]
    elif obsNumber == 3:
        freq = freqs[2][horizIndex]
    elif obsNumber == 4:
        freq = freqs[3][horizIndex]
    elif obsNumber == 5:
        freq = freqs[4][horizIndex]
    elif obsNumber == 6:
        freq = freqs[5][horizIndex]
    return(freq)

def reorganizingAllAsList(data, obsNumber):
    """
    Finding the time and freq for every data entry and putting them into a separate list. Input for function should
    be a waterfall data (refer to data list). horizDim corresponds to freq. vertDim corresponds to time.
    """
    plotTime = []
    plotFreqs = []
    dataAsList = []
    horizDim = data.shape[1]
    vertDim = data.shape[0]
    for i in range(vertDim):
        for j in range(horizDim):
            plotTime.append(locateTime(obsNumber, i))
            plotFreqs.append(locateFreq(obsNumber, j))
            dataAsList.append(data[i, j])
    return(plotFreqs, plotTime, dataAsList)

def reorganizingAllAsArray(data, obsNumber):
    """
    Finding the time and freq for every data entry and putting them into an array. Input for function should
    be a waterfall data (refer to data list). horizDim corresponds to freq. vertDim corresponds to time.
    """
    plotFreqs, plotTime, dataAsList = reorganizingAllAsList(data, obsNumber)
    horizDim = data.shape[1]
    vertDim = data.shape[0]
    dataAsArray = np.array(dataAsList).reshape(vertDim, horizDim)
    plotFreqsArray = np.array(plotFreqs).reshape(vertDim, horizDim)
    plotTimeArray = np.array(plotTime).reshape(vertDim, horizDim)
    return(plotFreqsArray, plotTimeArray, dataAsArray)

def ticksArray(obsNum):
    """
    Finding the frequency ticks for an observation number's data and putting them into an array.
    Input for function should be an observation number (1-6). horizDim corresponds to
    freq. vertDim corresponds to time. The frequency ticks will be used for plotting the ticks
    on the x axis.
    """
    dataObs = data[obsNum - 1]
    if type(args.freqstart) == float:
        freqsLin = np.around(np.linspace(args.freqstart, args.freqend, 6), 2)
        freqsLinTicks = np.around(np.linspace(freqsLin[0], freqsLin[-1], 5), 2)
    else:
        freqsArray,_,_ = reorganizingAllAsArray(dataObs, obsNum)
        freqsLin = np.around(np.linspace(np.amin(freqsArray), np.amax(freqsArray), 6), 2)
        freqsLinTicks = np.around(np.linspace(freqsLin[1], freqsLin[-1], 4), 2)
    return(freqsLinTicks)

numbers = [1, 2, 3, 4, 5, 6]
ticks = [ticksArray(obsNum) for obsNum in numbers]
def waterfall_plot_on(index):
    """
    Plotting the on observations waterfall plots. The if statements decide whether the plots
    will be focused on a freq/time range.
    """
    if type(args.freqstart) == float:
        plt.subplot(int('23' + str(index)), title = titles[2*(index - 1)], xticks = ticks[2*(index - 1)])
        waterfalls[2*(index - 1)].plot_waterfall(f_start = args.freqstart, f_stop = args.freqend)
        plt.title(titles[2*(index-1)])
        if type(args.tstart) == float:
            plt.ylim(args.tstart, args.tend)
    elif type(args.tstart) == float:
        plt.subplot(int('23' + str(index)), title = titles[2*(index - 1)], xticks = ticks[2*(index - 1)])
        waterfalls[2*(index - 1)].plot_waterfall()
        plt.ylim(args.tstart, args.tend)
        plt.title(titles[2*(index-1)])
    else:
        plt.subplot(int('23' + str(index)), title = titles[2*(index - 1)], xticks = ticks[2*(index - 1)])
        waterfalls[2*(index - 1)].plot_waterfall()
        plt.title(titles[2*(index-1)])

def waterfall_plot_off(index):
    """
    Plotting the off observations waterfall plots. The if statements decide whether the plots
    will be focused on a freq/time range.
    """
    if type(args.freqstart) == float:
        plt.subplot(int('23'+str(index)), title = titles[2*index - 7], xticks = ticks[2*index - 7])
        waterfall_off = waterfalls[2*index - 7]
        waterfall_off.plot_waterfall(f_start = args.freqstart, f_stop = args.freqend)
        plt.title(titles[2*index - 7])
        if type(args.tstart) == float:
            plt.ylim(args.tstart, args.tend)
    elif type(args.tstart) == float:
        plt.subplot(int('23'+str(index)), title = titles[2*index - 7], xticks = ticks[2*index - 7])
        waterfall_off = waterfalls[2*index - 7]
        waterfall_off.plot_waterfall()
        plt.ylim(args.tstart, args.tend)
        plt.title(titles[2*index - 7])
    else:
        plt.subplot(int('23'+str(index)), title = titles[2*index - 7], xticks = ticks[2*index - 7])
        waterfall_off = waterfalls[2*index - 7]
        waterfall_off.plot_waterfall()
        plt.title(titles[2*index - 7])

def Plot6Plots(saveAs):
    """
    Plotting all 6 Plots and saving it as pdf/png
    """
    plt.rc('text', usetex=True)
    plt.rc('font', size = 16, family='serif')
    plt.rc('axes', titlesize = 16, labelsize = 16)
    plt.rc('xtick', labelsize = 14)
    plt.rc('ytick', labelsize = 14)
    os.system('cls')
    plt.figure(figsize = (19, 10))
    [waterfall_plot_on(index) for index in [1, 2, 3]]
    [waterfall_plot_off(index) for index in [4, 5, 6]]
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    plt.show()
    if type(args.tstart) == float or type(args.freqstart) == float:
        if saveAs.lower() == 'png':
            plt.savefig(titles[0] + '_6plotsExtracted.png', dpi = 600)
        elif saveAs.lower() == 'pdf':
            plt.savefig(titles[0] + '_6plotsExtracted.pdf')
        else:
            pass
    else:
        if saveAs.lower() == 'png':
            plt.savefig(titles[0] + '_6plots.png', dpi = 600)
        elif saveAs.lower() == 'pdf':
            plt.savefig(titles[0] + '_6plots.pdf')
        else:
            pass

Plot6Plots(args.saveAs)
