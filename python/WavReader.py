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
        
        # Read in wav file
        self.sampleRate, self.data = wavfile.read(filepath)

        # Get number of samples and channels
        self.numSamples = np.size(self.data,0)  # 0th dimension: Samples
        self.numChannels = np.size(self.data,1) # 1th dimension: Channels

    ## Plots the wav file
    def plot(self):

        # Create time vector
        time = np.arange(0, self.numSamples/self.sampleRate, 1/self.sampleRate)

        # Plot wavfile data vs time
        plt.figure(1)

        # create sub plot number based on number of channels
        spn = self.numChannels*100 + 11 

        # Loop through all channels (transposing data)
        for chan in self.data.T: 
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

