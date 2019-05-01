import argparse
from zipfile import ZipFile
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import h5py
import blimpy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
parser = argparse.ArgumentParser(description='Enter a filepath for a zipfile, the mode of file you want to save plots, and on observation number.')
"""Adding arguments"""
parser.add_argument(
    'zipfilepath',
    help = 'The filepath of the zipfile representing observation cadence',
    type = str,
)
parser.add_argument(
    'onObsNumber',
    help = 'The on observation number of the cadence you wish to analyze along with its off observation',
    type = int,
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

filepathZip = args.zipfilepath
def read_zip_file(filepath):
    """
    Unzips a zip file given the filepath of the zip. If filepath is a zip and the function finds that the
    unzipped version already exists, will return 'This file has already been unzipped'.
    """
    zip1title = filepath[0:-4]
    cwd = os.getcwd()
    pdir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if os.path.isdir(zip1title) == True:
        print('This file has already been unzipped.')
        nameList = []
        for file in os.listdir(zip1title):
            file = zip1title + '/' + file
            nameList.append(file)
        return(nameList)
    elif os.path.isfile(filepath) == False and filepath[-4:] == '.zip':
        print("The zipfile you entered isn't in the previous directory with respect to your current working directory: " + cwd)
        print("Please check that you've downloaded the file and move the zip file from the Downloads section (or wherever you saved it) to the parent directory: " + pdir)
    elif filepath[-4:] != '.zip':
        print("The file you entered is not a zipfile. Please enter a zipfile.")
    elif os.path.isfile(filepath) == True:
        zip1 = ZipFile(filepath)
        nameList = zip1.namelist()
        newNameList = []
        for name in nameList:
            name = '../' + name
            newNameList.append(name)
        zip1.extractall('../')
        zip1.close()
        os.remove(filepath)
        return(newNameList)

observationFilepaths = read_zip_file(filepathZip)

def h5_header_wrapper(filename):
    """
    Gets header of a .h5 file.
    """
    h = h5py.File(filename)
    header = dict(h['data'].attrs.items())
    return header

def waterfallfunc(filepath):
    """
    Reads .h5 filepath into waterfall filterbank data
    """
    return(blimpy.Waterfall(filepath))

waterfalls = [waterfallfunc(filepath) for filepath in observationFilepaths]
hd = [h5_header_wrapper(filepath) for filepath in observationFilepaths]
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
    freqsArray,_,_ = reorganizingAllAsArray(dataObs, obsNum)
    freqsLin = np.around(np.linspace(np.amin(freqsArray), np.amax(freqsArray), 6), 2)
    freqsLinTicks = np.around(np.linspace(freqsLin[1], freqsLin[-1], 4), 2)
    return(freqsLinTicks)

numbers = [1, 2, 3, 4, 5, 6]
ticks = [ticksArray(obsNum) for obsNum in numbers]

def PlotWithOffObs(onObsNumber, saveAs):
    """
    Plotting the waterfall plots for the on observation chosen as well as the off observation. onObsNumber are 1, 3, or 5.
    Also, the argument saveAs will decide what to save the plots as. saveAs will have 3 options for input: pdf, png, None.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', size = 16, family='serif')
    plt.rc('axes', titlesize = 16, labelsize = 16)
    plt.rc('xtick', labelsize = 15)
    plt.rc('ytick', labelsize = 15)
    os.system('cls')
    titleOn, titleOff = titles[onObsNumber - 1], titles[onObsNumber]
    waterfallOn, waterfallOff = waterfalls[onObsNumber - 1], waterfalls[onObsNumber]
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (9, 14))
    """onObservation plot"""
    axes1 = plt.subplot(211, xticks = ticks[onObsNumber - 1], title = titleOn + " On Observation " + str(onObsNumber))
    waterfallOn.plot_waterfall()
    axes1.set_title(titleOn + " On Observation " + str(onObsNumber))
    """offObservation plot"""
    axes2 = plt.subplot(212, xticks = ticks[onObsNumber], title =  titleOff + ", Off Observation")
    waterfallOff.plot_waterfall()
    axes2.set_title(titleOn + " Off Observation " + str(onObsNumber))
    fig.tight_layout()
    plt.show()
    if saveAs == 'pdf':
        fig.savefig(titles[0] + '_onObservation' + str(onObsNumber) + '.pdf')
    elif saveAs == 'png':
        fig.savefig(titles[0] + '_onObservation' + str(onObsNumber) + '.png', dpi = 600)
    else:
        pass

def PlotWithZoom(onObsNumber, freqstart, freqend, tstart, tend, saveAs):
    """
    Plotting the waterfall plots for the on observation chosen as well as the off observation. onObsNumber are 1, 3, or 5.
    Freqstart, freqend, tstart, tend decide which section of the graph will be zoomed in on.
    saveAs will have 3 options for input: pdf, png, None.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', size = 16, family='serif')
    plt.rc('axes', titlesize = 16, labelsize = 16)
    plt.rc('xtick', labelsize = 15)
    plt.rc('ytick', labelsize = 15)
    os.system('cls')
    titleOn, titleOff = titles[onObsNumber - 1], titles[onObsNumber]
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 11))
    waterfallOn, waterfallOff = waterfalls[onObsNumber - 1], waterfalls[onObsNumber]
    """onObservation plot"""
    axes1 = plt.subplot(221, xticks = ticks[onObsNumber - 1], title = titleOn + " On Observation " + str(onObsNumber))
    waterfallOn.plot_waterfall()
    axes1.set_title(titleOn + " On Observation " + str(onObsNumber))
    """Zooming in for on"""
    axesOnObs = zoomed_inset_axes(axes1, 3, loc=1)
    waterfallOn.plot_waterfall(cb = False)
    axesOnObs.set_xlim(float(freqstart), float(freqend)) # apply the x-limits
    axesOnObs.set_ylim(float(tstart), float(tend)) # apply the y-limits
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    plt.xlabel('', visible=False)
    plt.ylabel('', visible=False)
    plt.title('', visible=False)
    mark_inset(axes1, axesOnObs, loc1=1, loc2=2, fc="none", ec="r")
    axes1zoom = plt.subplot(222)
    waterfallOn.plot_waterfall()
    axes1zoom.set_xticks(np.linspace(float(freqstart), float(freqend), 5, endpoint = True))
    axes1zoom.set_xlim([float(freqstart), float(freqend)])
    axes1zoom.set_ylim([float(tstart), float(tend)])
    axes1zoom.set_xlabel('Frequency [MHz]')
    axes1zoom.set_ylabel('Time [s]')
    axes1zoom.set_title("On Observation Zoomed In")
    """offObservation plot"""
    axes2 = plt.subplot(223, xticks = ticks[onObsNumber], title = titleOff + ", Off Observation")
    waterfallOff.plot_waterfall()
    axes2.set_title(titleOff + ", Off Observation")
    """Zooming in for off"""
    axesOffObs = zoomed_inset_axes(axes2, 3, loc=1)
    waterfallOff.plot_waterfall(cb = False)
    axesOffObs.set_xlim(float(freqstart), float(freqend))
    axesOffObs.set_ylim(float(tstart), float(tend))
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    plt.xlabel('', visible=False)
    plt.ylabel('', visible=False)
    plt.title('', visible=False)
    mark_inset(axes2, axesOffObs, loc1=1, loc2=2, fc="none", ec="r")
    axes2zoom = plt.subplot(224)
    waterfallOff.plot_waterfall()
    axes2zoom.set_xticks(np.linspace(float(freqstart), float(freqend), 5, endpoint = True))
    axes2zoom.set_xlim([float(freqstart), float(freqend)])
    axes2zoom.set_ylim([float(tstart), float(tend)])
    axes2zoom.set_xlabel('Frequency [MHz]')
    axes2zoom.set_ylabel('Time [s]')
    axes2zoom.set_title("Off Observation Zoomed In")
    fig.show()
    fig.tight_layout()
    if saveAs == 'pdf':
        fig.savefig(titles[0] + '_onObservation' + str(onObsNumber) + 'zoom.pdf')
    elif saveAs == 'png':
        fig.savefig(titles[0] + '_onObservation' + str(onObsNumber) + 'zoom.png', dpi = 600)
    else:
        pass

if type(args.freqstart) == float and type(args.tstart) == float:
    PlotWithZoom(args.onObsNumber, args.freqstart, args.freqend, args.tstart, args.tend, args.saveAs)
else:
    PlotWithOffObs(args.onObsNumber, args.saveAs)
