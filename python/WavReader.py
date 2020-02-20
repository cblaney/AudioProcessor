## 
# @package  WavReader
# 
# @author   Chris Blaney
#

import matplotlib.pyplot as plt
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
        self.sampleRate, self.data = wavfile.read(filepath)

        # Samples are the 0th dimension, Channels are the 1th dimension
        self.numSamples = np.size(self.data,0)
        self.numChannels = np.size(self.data,1)

    ## Plots the wav file
    def plot(self):
        # Create time vector
        time = np.arange(0, self.numSamples/self.sampleRate, 1/self.sampleRate)

        # Plot wavfile data vs time
        plt.figure(1)
        spn = self.numChannels*100 + 11    # create sub plot number based on number of channels
        for chan in self.data.T:         # Loop through all channels (transposing data)
            plt.subplot(spn)        # Create subplot
            plt.plot(time, chan)    # plot the data
            plt.xlabel('time (s)')
            plt.ylabel('amplitude')
            spn = spn+1             # increase subplot number
        
        # display wavfile
        plt.show()

    
# Test code
wr = WavReader('./../audio/c_chord.wav')
wr.plot()

