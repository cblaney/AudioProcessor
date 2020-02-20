## 
# @package  WavReader
# 
# @author   Chris Blaney
#

import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np
from scipy.io import wavfile

## Object to read in wav files
# 
# Allows easy access to wav file data
#
class WavReader:

    ## Initializer
    #
    # Reads in a given wav file
    #
    def __init__(self, filepath):
        
        # Read in wav file
        self.sampleRate, self.data = wavfile.read(filepath)

        # Get number of samples and channels
        self.numSamples = np.size(self.data,0)  # 0th dimension: Samples
        self.numChannels = np.size(self.data,1) # 1th dimension: Channels


    ## Returns wav data as single channel audio
    def getMono(self):
        
        # return the normalized sum across the channels
        return np.sum(self.data, axis=1)/self.numChannels


    ## Plots the wav file
    def plot(self):

        # Create time vector
        time = np.arange(0, self.numSamples/self.sampleRate, 1/self.sampleRate)

        # Create a new figure
        plt.figure()

        # create sub plot number based on number of channels
        spn = self.numChannels*100 + 11 

        # Loop through all channels (transposing data)
        for chan in self.data.T: 
            plt.subplot(spn)        # Create subplot
            plt.plot(time, chan)    # plot the data
            plt.xlabel('time (s)')
            plt.ylabel('amplitude')
            spn = spn+1             # increase subplot number
        
    
# Test code
interactive(True)
wr = WavReader('./../audio/c_chord.wav')
plt.figure()
plt.plot(wr.getMono())
wr.plot()
input('press return to continue')

