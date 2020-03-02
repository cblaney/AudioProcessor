##
#
#
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# Notation Symobls:
#   Mathematic:
#   ∈ "is an element of" {alt+8712}
#   ∆ "delta" {alt+8710}
#   ∑ "sum" {alt+8721}
#   Ψ "Psi" {alt+936} ( Principle argument function ensures that
#                       the input lies within range [-0.5, 0.5] )
#   Greek:
#   π "pi" {alt+960}
#   φ "phi" {alt+966}
#   ω "omega" {alt+969}
# 
#


# 
# Discrete STFT (Short-Time Fourier-Transform):
# n is time[]
# k is freq[]
# x[] is out input signal
# w[0:N-1] is a window of length N
# N is window size and fft size
# H is the hopping size parameter
#
# (8.2)  Xx(n,k) = SUM_{r=0}^{N-1} ( x(r+nH) w(r) exp(-j2πkr/N) )
#
# (8.2a) Xx(n,k) = FFT( x[nH:nH+N-1] w[:] ) 
#
# (8.2b) Xx(n,k) = {k: X[k] = FFT( x[nH:nH+N-1] w[:] ) }
#
def STFT(x, Fs, N, H, real=True):
    F_coef, T_coef, Xx = signal.stft(x, Fs, nperseg=N, noverlap=N-H, return_onesided=real)
    return T_coef, F_coef, Xx.T

def iSTFT(Xx, Fs, N, H):
    return signal.istft(Xx.T, Fs, nperseg=N, noverlap=N-H, input_onesided=True)


## Returns the pitch bin index for given frequency
#
# Bin(ω) = floor(12 log_2(ω/440)+69.5)
# or
# Bin(ω) = floor(1200/R log_2(ω/ω_ref) + 1.5/12)
# where
#   ω_ref is a reference frequency assigned to bin index 1
#   R is the desired resolution of log spaced freq axis (cents)
#
def BinIndex(w, R=100, w_ref=8):
    return np.floor(((1200/R) * np.log2(unzero(w)/w_ref)) + 0.125)


def unzero( a):
        b=np.array(a,dtype=np.float)
        b[b==0]=1e-14
        return b

## Returns the set of F_coef indexes for pitch bin b
#
# P(p) = {k: Bin(F_coef(k)) = p}
# or
# P(b) = {k: Bin(F_coef(k)) = b}
#
def PitchIndexes(b, F_coef, R=100, w_ref=8):
    return np.where(BinIndex(F_coef,R,w_ref)==b)[0]

#
#
# if B(ω) = floor(1200/R log_2(ω/ω_ref) + 1.5/12)
# then
# F(b) = w_ref * 2^((b-4.5/12)R/1200)
def BinFrequency(b, R=100, w_ref=8):
    return w_ref * (2 ** ((b-0.375)*R/1200))
    

## Returns a log-frequency spectrogram
#
# Refined Log-Frequency Spectrogram:
#
# Bin(ω) = floor(1200/R log_2(ω/ω_ref) + 1.5/12)
#
# P(b) = {k: Bin(F_coef(k)) = b}
#
# Yy_LF(n,b) = SUM_{k ∈ P(b)} ( |Xx(n,k)|^2 )
#
def LogFreqSpectrogram(Xx, F_coef, N_p=128, R=100, w_ref=8):
    
    Xx_Log=np.zeros((Xx.shape[0], N_p))
    
    ## for Xn_log,Xn in zip(Xx_Log, Xx):
    ##     pitches, Xn_log = LogFreqSpectral(Xn,F_coef,N_p,R,w_ref)
    for i in np.arange(Xx.shape[0]):
        pitches, Xx_Log[i] = LogFreqSpectral(Xx[i],F_coef,N_p,R,w_ref)

    return pitches, Xx_Log

## Returns a log-frequency spectral
#
# Y_LF(b) = SUM_{k ∈ P(b)} ( |X(k)|^2 )
#
def LogFreqSpectral(X, F_coef, N_p=128, R=100, w_ref=8):

    # get pitches    
    pitches=np.arange(N_p)

    # allocate log-freq spectral array
    X_Log=np.zeros(N_p)
    
    # Compute log-freq spec
    # for each b in X_log, sum X(P(b))^2
    ## for b,p in zip(X_Log,pitches):
    ##     b = sum( X[PitchIndexes(p,F_coef,R,w_ref)] ** 2 )
    for i in np.arange(X_Log.shape[0]):
        X_Log[i] = sum( abs(X[PitchIndexes(pitches[i],F_coef,R,w_ref)]) ** 2 )

    return pitches, X_Log

#
# Instantaneous Freq
#
# Polar Representation of STFT
# Xx(n,k) = |Xx(n,k)| exp(j2π φ(n,k))
# where:
#   φ(n,k) ∈ [0,1)
#
# Radial Frequency
# ω = F_coef(k) = kFs/N
# where:
#   k ∈ [0:N/2]
#
# Two Time instances
# t1 = T_coef(n-1) = (n-1)H/Fs
# t2 = T_coef(n)   = nH/Fs
#
# Measured Phases at Time instances
# φ1 = φ(n-1,k)
# φ2 = φ(n,k)
#
# Instantaneous Freq at t1:
# ω_t1 = lim_{t2->t1} ( (φ2-φ1)/(t2-t1) )
# or
# ω = lim_{t2->t1} ( ∆φ/∆t )
# 
# Phase prediction and error
# φ_Pred = φ1 + ω∆t
# φ_Err = Ψ(φ2 - φ_Pred)
# 
# Refined Instantanous Freq estimate
# IF(ω) = ω + φ_Err/∆t
#       = ω + Ψ(φ2-(φ1+ω∆t))/∆t
#
# Instantaneous Frequency Coef
# IF_coef(k,n) = IF(ω)
#              = ω + Ψ(φ2-(φ1+ω∆t))/∆t
#              = kFs/N + Ψ(φ(n,k)-(φ(n-1,k)-kFs/N(H/Fs)))/(H/Fs)
#
# IF_coef(k,n) = (k + K(k,n))Fs/N
#   where K(k,n) is the Bin offset
#   K(k,n) = Ψ( φ(n,k)-φ(n-1,k)-kH/N )N/H
#
# IF_coef(k,n) = (k + (Ψ( φ(n,k)-φ(n-1,k)-kH/N )N/H) )Fs/N
#
def InstFreqCoef(Xx, Fs, N, H):

    # get F_Coef indexes
    k = np.arange(Xx.shape[1])
    
    # get φ1 and φ2
    p1 = np.angle(Xx)/np.pi
    p2 = np.append(p1[1:],np.zeros((1,p1.shape[1])), axis=0)

    # Compute Ψ(K)
    K = np.modf(p2-p1-(k*H/N))[0]   # Get the modulo 1 of K
    i = np.where(K>0.5)             # Get the indexes K > 0.5
    K[i]=K[i]-1                     # Subtract indexes by 1

    # Compute and return IF_Coef
    return (Fs / N) * (k + K)

#
# Using Instantanous Freq
#   Instead of taking the center freq F_coef(k), employ the refined 
#   freq estimates F_IF_coef(k,n) for the defining the sets
# 
# P_IF(b,n) = {k: Bin(IF_coef(k,n) = b}
#
# Y_ILF(n,b) = SUM_{k ∈ P_IF(b,n)} ( |Xx(n,k)|^2 )
#
def InstLogFreqSpectrogram(Xx, Fs, N, H, F_coef, N_p=128, R=100, w_ref=8):
    IF_coef = InstFreqCoef(Xx, Fs, N, H)

    Xx_Log=np.zeros((Xx.shape[0], N_p))

    ## for Xn_log,Xn in zip(Xx_Log, Xx):
    ##     pitches, Xn_log = LogFreqSpectral(Xn,F_coef,N_p,R,w_ref)
    for i in np.arange(Xx.shape[0]):
        pitches, Xx_Log[i] = LogFreqSpectral(Xx[i],IF_coef[i],N_p,R,w_ref)

    return pitches, Xx_Log

    
#
# Discrete Instantaneous Log Frequency STFT (Short-Time Fourier-Transform):
#
# (8.2b) Xx(n,k) = {k: X[k] = FFT( x[nH:nH+N-1] w[:] ) }
#
# Y_ILF(n,b) = SUM_{k ∈ P_IF(b,n)} ( |Xx(n,k)|^2 )
#
def ILFSTFT(x, Fs, N, H, N_p=128, R=100, w_ref=8):
    T_coef, F_coef, Xx = STFT(x, Fs, N, H)
    P, Y_ILF = InstLogFreqSpectrogram(Xx, Fs, N, H, F_coef, N_p,R,w_ref)
    return T_coef, P, Y_ILF
    

#
# Harmonic Summation
#
# For the Spectrogram Y: 
# Y_H(n,k) = SUM_{h=1}^{H} ( Y(n,kh) )
#
# For Log Frequency Spectrogram Y_LF:
# Y_HLF(n,b) = SUM_{h=1}^{H} ( Y_LF(n, b + floor(log2(h)1200/R) ) )
def HarmonicSum(X, H=4):
    
    # Create zero padded spectral
    X_z = np.append(X, np.zeros(len(X)*(H-1)))

    X_H = X.copy()
 
        


# Testing code
if (__name__ == "__main__"):
    # Testing tools
    from matplotlib import interactive
    interactive(True)

    # Local Tools
    import WavReader

    # make a wavreader and collect mono audio
    #wr = WavReader.WavReader('./../audio/C Chord - 1.3 - Acoustic Piano.wav')
    wr = WavReader.WavReader('/mnt/d/Sheet Music/Guitar Bass/Audio Testing/BtBaM - Mirrors.wav')
    #wr = WavReader.WavReader('/mnt/d/Music/The Great Misdirect (Vinyl Remaster)/01 Mirrors.wav')

    sig = wr.getMono()
    #sig = sig[0:int(len(sig)/3)]
    plt.figure()
    plt.plot(sig)

    # # Original Spectrogram
    # T, F, Spec = STFT(sig, wr.sampleRate, 2**13, 2**12)
    # plt.figure()
    # plt.pcolormesh(T, F, abs(Spec.T))

   
    # # Log Frequency Spectrogram
    # P, LFSpec = LogFreqSpectrogram(Spec, F)
    # plt.figure()
    # plt.pcolormesh(T, P, abs(LFSpec.T))

    # # Instantaneous Log Frequency Spectrogram
    # P_IF, ILFSpec = InstLogFreqSpectrogram(Spec, wr.sampleRate, 2**13, 2**12, F)
    # plt.figure()
    # plt.pcolormesh(T, P_IF, abs(ILFSpec.T))

    # ILFSTFT  
    T2, P2, ILFSpec2 = ILFSTFT(sig, wr.sampleRate, 2**13, 2**12, N_p=256, R=50)
    plt.figure()
    plt.pcolormesh(T2, P2, np.sqrt(abs(ILFSpec2.T)))


    input('press return to continue')

