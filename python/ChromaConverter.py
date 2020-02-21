##
# @package  ChromaConverter
#
# @author   Chris Blaney
#

import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np

## Object to create Chroma representations of signal data
#
class ChromaConverter:

    ## Initializer
    #
    # Setup ChromaConverter by calculating frequency spectrum and pitches
    #
    def __init__(self, fs, P_0=0, P_N=120, dev=1, A4_Hz=440, A4_idx=69):
        self.sampleFreq = fs
        self.dev = dev
        self.pitch_idx = np.arange(P_0,P_N)
        self.pitch = self.getPitchFreq(self.pitch_idx,A4_Hz,A4_idx)
        self.freq = np.arange(0, self.sampleFreq, self.sampleFreq/int(len(sig)/2-1))
   
    ## Returns Chroma for input time domain signal
    #
    def getChroma(self, sig):
        
        # check size
        if(self.freq.size != sig.size):
            self.freq = np.arange(0, self.sampleFreq, self.sampleFreq/int(len(sig)/2-1))
        
        # Take DFT
        spec = np.fft.fft(sig)
        spec = spec[0:int(len(spec)/2-1)]
    
        # Create Chroma
        return self.calculateChroma(spec, self.freq, self.pitch, self.dev)
    
    ## Calculates chroma
    #
    # Y(n) = SUM_0<=w<=W( { P(n-0.5) <= w < P(n+0.5): X(w) } )
    def calculateChroma(self, spec, freq, pitch, dev):
        chroma = np.arange(0, np.size(pitch,0))
        for i in chroma:
            chroma[i] = abs(self.getSumBins([pitch[i]-dev*0.5,pitch[i]+dev*0.5], spec, freq))
        return chroma;
    
    ## Aggregates bin values between two frequencies
    #
    # SUM_0<=w<=W( { W_0 <= w < W_1: X(w) } )
    def getSumBins(self, win, spec, freq):
        return sum( spec[ (win[0] <= freq) & (freq < win[1]) ] )
    
    ## Calculates frequency of pitch at index
    #
    def getPitchFreq(self, idx, A4_Hz=440, A4_idx=69):
        return 2 ** ((idx - A4_idx)/12) * A4_Hz

# Testing code
if (__name__ == "__main__"):
    import WavReader
    interactive(True)
    
    # Read in wav and get mono data
    wr = WavReader.WavReader('./../audio/c_chord.wav')
    sig = wr.getMono()

    plt.figure()
    plt.plot(sig)

    cc = ChromaConverter(wr.sampleRate)
    chroma = cc.getChroma(sig)

    plt.figure()
    plt.imshow( [ cc.pitch_idx, chroma ] )

    input('press return to continue')

