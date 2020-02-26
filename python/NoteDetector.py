##
# @package  NoteDetector
#
# @author   Chris Blaney
#

import matplotlib.pyplot as plt
from matplotlib import interactive
import numpy as np

## Returns the number of overtones for a given pitch and pitch maxima vector
def getNumberOvertones(pitch, maxima):
    overtones = 0;
    while ( (maxima == getOvertonePitch(overtones, pitch)).any() ):
        overtones += 1
    return overtones

## Returns harmonic pitch 
# 
# O = 12*log2(ov+1)
#
def getOvertonePitch(ov, pitch):
    return round(12*np.log2(ov+1) + pitch)
    

def getMaxima(arr):
    # a[n] > a[n+1]  &  a[n] > a[n-1]
    maxima = np.r_[(arr[:-1] > arr[1:]), True] & \
             np.r_[True,(arr[1:] > arr[0:-1])]
    maxima_val = arr[maxima]
    maxima_idx = np.where(maxima)[0] 

    return maxima_val, maxima_idx

# Testing Code
if (__name__ == "__main__"):
    import WavReader
    import ChromaConverter
    interactive(True)

    # Read in wav and get mono data
    wr = WavReader.WavReader('./../audio/C Chord - 1.3 - Acoustic Piano.wav')
    sig = wr.getMono()
    # Get Chromagram
    cc = ChromaConverter.ChromaConverter(wr.sampleRate)
    indexes, chromagram = cc.getChromagram(sig)
    # plot chromagram
    plt.figure()
    plt.imshow(chromagram.T, aspect='auto')
    print(indexes)

    # find notes!
   
    #for c in chromagram: 
   
    c = chromagram[48]
 
    # a[n] > a[n+1]  &  a[n] > a[n-1]
    m1_val, m1_idx = getMaxima(c)

    m2_val, m2_idx = getMaxima(m1_val)
    m2_idx = m1_idx[m2_idx]

    m3_val, m3_idx = getMaxima(m2_val)
    m3_idx = m2_idx[m3_idx]

    plt.figure()
    plt.plot(np.arange(len(c)), c)
    plt.plot(m1_idx, m1_val)
    plt.plot(m2_idx, m2_val)
    plt.plot(m3_idx, m3_val)
    
    c_1 = c.copy();
    mask = np.ones(c_1.shape, dtype=bool)
    mask[m2_idx] = 0;
    c_1[mask] = 0;

    plt.figure()
    plt.plot(c_1) 

    print(cc.getPitchFreq(m2_idx))
#    print(2, getOvertonePitch(2,m3_idx[0]), (m1_idx == getOvertonePitch(2,m3_idx[0])).any() )

    roots = []
    for r in m1_idx:
        numOv = getNumberOvertones(r, m1_idx)
        if(numOv >= 2):
            print(r, numOv)

    


    

#    for m in m3_idx:
#        root_pitch = m
#        harm_1 = getOvertonePitch(1,m)
#        
        


#    maxima_1 = np.r_[(c[:-1] > c[1:]), True] & \
#             np.r_[True,(c[1:] > c[0:-1])]
#    maxima_1_val = c[maxima_1]
#    maxima_1_idx = np.where(maxima_1)[0]
#
#    m = maxima_1_val
#    
#    maxima_2 = np.r_[(m[:-1] > m[1:]), True] & \
#             np.r_[True,(m[1:] > m[0:-1])]
#    
#    maxima_2_val = m[maxima_2]
#    maxima_2_idx = maxima_1_idx[np.where(maxima_2)[0]]



    

     
    
    
    input('press return to continue')
    

    
