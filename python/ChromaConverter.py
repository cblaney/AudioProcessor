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
    # A0_idx=21
    def __init__(self, fs, P_0=0, P_N=120, dev=1, A4_Hz=440, A4_idx=69):
        self.sampleFreq = fs
        self.dev = dev
        self.pitch_idx = np.arange(P_0,P_N)
        self.pitch = self.getPitchFreq(self.pitch_idx,A4_Hz,A4_idx)
        self.freq = np.arange(0, 1)

    
    ## Computes chromagram
    #
    def getChromagram(self, sig, fftsize=(2**19), windowsize=(2**13), overlap=int(2**12.5)):
        
        # create window
        win = np.hamming(windowsize)

        # calculate indexes
        indexes = np.arange(0, len(sig)-1-windowsize, windowsize-overlap)
        
        # loop and calculate singleChroma
        chromagram = []
        for i in indexes:
            if(np.amax(sig[i:i+windowsize]) > 0):
                windowed = np.pad(sig[i:i+windowsize]*win,(0,fftsize-windowsize%fftsize))
                chromagram.append(self.getSingleChroma(windowed))
            else:
                chromagram.append(np.zeros(np.size(self.pitch,0)))
            
        return indexes,np.array(chromagram)
   
    ## Returns Chroma for input time domain signal
    #
    def getSingleChroma(self, sig):
        
        # check size
        if(len(self.freq) != int(len(sig)/2)):
            print('Changing size from ' + repr(self.freq.size) + ' to ' + repr(int(sig.size/2)))
            self.freq = np.arange(0, self.sampleFreq, self.sampleFreq/int((len(sig))/2))
        
        # Take DFT
        spec = np.fft.fft(sig)
        spec = spec[0:int((len(spec))/2)]

        # Create Chroma
        return self.calculateChroma(abs(spec)/len(sig), self.freq, self.pitch, self.dev)
    
    ## Calculates chroma
    #
    # Y(n) = SUM_0<=w<=W( { P(n-0.5) <= w < P(n+0.5): X(w) } )
    def calculateChroma(self, spec, freq, pitch, dev):
        chroma = []
        for p in pitch:
            chroma.append(self.getSumBins([p-dev*0.5, p+dev*0.5], spec, freq))
        # perform Logarithmic compression
        return np.log(1+10*self.unzero(chroma));
    
    ## Aggregates bin values between two frequencies
    #
    # SUM_0<=w<=W( { W_0 <= w < W_1: X(w) } )
    def getSumBins(self, win, spec, freq):
        return sum( (spec[ (win[0] <= freq) & (freq < win[1]) ] )**2 )
    
    ## Calculates frequency of pitch at index
    #
    # f_A4=440(p) = 2^(p-69)/12 * 440
    def getPitchFreq(self, idx, A4_Hz=440, A4_idx=69):
        return 2 ** ((idx - A4_idx)/12) * A4_Hz

    ##
    def unzero(self, a):
        b=np.array(a,dtype=np.float)
        b[b==0]=1e-10
        return b
    

# Testing code
if (__name__ == "__main__"):
    import WavReader
    interactive(True)
    
    # Read in wav and get mono data
    wr = WavReader.WavReader('./../audio/C Chord - 1.3 - Acoustic Piano.wav')
#    wr = WavReader.WavReader('/mnt/d/Sheet Music/Guitar Bass/Audio Testing/BtBaM - Mirrors.wav')
#    wr = WavReader.WavReader('/mnt/d/Music/The Great Misdirect (Vinyl Remaster)/01 Mirrors.wav')
    sig = wr.getMono()

    plt.figure()
    plt.plot(sig)

    cc = ChromaConverter(wr.sampleRate)
    chroma = cc.getSingleChroma(sig)

    plt.figure()
    plt.imshow( [ cc.pitch_idx, chroma ] )

    plt.figure()
    plt.plot(cc.pitch_idx, chroma)

    plt.figure()
    indexes, chromagram = cc.getChromagram(sig)

    plt.imshow(chromagram.T, aspect='auto')

    plt.figure()
    plt.plot(chromagram[60])

    input('press return to continue')

