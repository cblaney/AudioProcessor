#
#
#
import matplotlib.pyplot as plt
import numpy as np

## Notation Symobls:
#   Mathematic:
#
#   ℕ "Natrual Numbers" {alt+8469}
#   ℝ "Real Numbers" {alt+8477}
#   ∂ "italic delta: Partial Derivative of" {alt+8706}
#   ∈ "is an element of" {alt+8712}
#   ∆ "Delta: Change of" {alt+8710}
#   ∇ "nabla: differential operator" {alt+8711}
#   ∑ "Sigma: Sum of" {alt+8721}
#   ∙ "Matrix dot product" {alt+8729}
#   ∫ "Integral of" {alt+8747}
#   ≈ "Approximate Equality" {alt+8776}
#
#   Greek:
#   Ψ "Psi" {alt+936} ( Principle argument function ensures that
#                       the input lies within range [-0.5, 0.5] )
#   γ "gamma" {alt+947}
#   δ "delta" {alt+948}
#   ε "epsilon" {alt+949}
#   μ "mu" {alt+956}
#   π "pi" {alt+960}
#   φ "phi" {alt+966}
#   ω "omega" {alt+969}
#
#   Miscellaneous:
#    Extended Symbols:
#     ( {alt+9115:9117}
#     ) {alt+9118:9120}
#     [ {alt+9121:9123}
#     ] {alt+9124:9126}
#     { {alt+9127:9130}
#     } {alt+9131:9134}


##
#
#
#   -> STFT -> Horz/Vert MedianFiltering -> Binary Masking -> iSTFT ->
#
# Let x be a discrete time audio signal with harmonic compenent signal x^h 
# and percussive component signal x^p:
#
#   x = x^h + x^p
#
# Compute discrete STFT Xx: (Spectral.py)
#   Xx(n,k) = {k: X[k] = FFT( x[nH:nH+N-1] w[:] ) }
#
# Derive the (power) spectrogram Y
#   Y(n,k) = |X(n,k)|^2
#

## Median Filtering
# 
# Median of A: μ1/2(A)
#
# Median is applied to a neighborhood of elements of length L
# to define the concept of median filtering:
#
#   μ1/2^L[A](n) = μ1/2( a_(n-(L-1)/2), ... , a_(n+(L-1)/2) )
#
def MedianFilter (A, L):
    # assume L is odd
    if (np.mod(L,2) == 0):
        print('Odd L needed, adding by 1')
        L = L+1

    # zero pad
    Az = np.append(np.zeros(int(L/2)), np.append(A, np.zeros( int(L/2) )) )  
    
    # pre allocate npArray
    Am = np.zeros(len(A))

    # median filter
    for i in np.arange(Am.shape[0]):
        Am[i] = np.median(Az[i:i+L])

    return Am  

def MedianFilterSpectral(Y,L):
    # assume L is odd
    if (np.mod(L,2) == 0):
        print('Odd L needed, adding by 1')
        L = L+1

    # pre allocate
    Ym = np.zeros(Y.shape)
    
    # filter
    for i in np.arange(Ym.shape[0]):
        Ym[i] = MedianFilter(Y[i],L)

    return Ym
    
##
# Horizontal filtered with filtering parameter L^h
# 
#   Y^h(n,k) = μ1/2( Y(n-(L^h-1)/2,k), ... , Y(n+(L^h-1)/2,k) )
#
def HorzMedianFilter(Y, Lh):
    # Transpose for horiz
    return MedianFilterSpectral(Y.T,Lh).T

##
# Vertical filtered with filtering parameter L^p
#
#   Y^p(n,k) = μ1/2( Y(n,k-(L^p-1)/2), ... , Y(n,k+(L^p-1)/2) )
#
def VertMedianFilter(Y, Lp):
    return MedianFilterSpectral(Y,Lp)


## Binary Masking
#
#             ⎧ 1, Y^h(n,k) >= Y^p(n,k)
# M^h(n,k) := ⎨ 
#             ⎩ 0, otherwise
#
#             ⎧ 1, Y^h(n,k) < Y^p(n,k)
# M^p(n,k) := ⎨ 
#             ⎩ 0, otherwise
#
def BinaryMask(Yh, Yp):
    Mh = (Yh >= Yp)
    return np.array(Mh,dtype=int), np.array(~Mh,dtype=int)

## Soft Masking
#
#   M^h(n,k) := (Y^h(n,k) + ε/2) / (Y^h(n,k) + Y^p(n,k) + ε)
#   M^p(n,k) := (Y^p(n,k) + ε/2) / (Y^h(n,k) + Y^p(n,k) + ε)
#   Where ε is a small positive value added to avoid division by zero
#
def SoftMask(Yh, Yp):
    e = 1e-14
    div = 1 / (Yh + Yp + e)
    Mh = (Yh + e/2) * div
    Mp = (Yp + e/2) * div
    return Mh, Mp


# Testing code
if(__name__ == "__main__"):
    # Testing tools
    from matplotlib import interactive
    interactive(True)

    # Local tools
    import WavReader
    import Spectral

    # Collect mono audio
#   wr = WavReader.WavReader('./../audio/C Chord - 1.3 - Acoustic Piano.wav')
    wr = WavReader.WavReader('./../audio/Harmonic&Percussive - 1.2 - Violin.wav')
    xh_o = wr.getMono()
    Fs = wr.sampleRate
    xp_o = WavReader.WavReader('./../audio/Harmonic&Percussive - 2.2 - Drumkit.wav').getMono()
    x = xh_o + xp_o
    t = np.arange(len(x))/Fs
    plt.figure()
    plt.title('Audio Signal')
    plt.plot(t,x)
    plt.figure()
    plt.subplot(211)
    plt.title('Original Harmonic Audio Signal')
    plt.plot(xh_o)
    plt.subplot(212)
    plt.title('Original Percussive Audio Signal')
    plt.plot(xp_o)

    # Power Spectrogram
    print('Calculating Spectrogram')
    T, F, Xx = Spectral.STFT(x, Fs, 2**13, 2**12)
    Y = (abs(Xx) ** 2)
    #Y = abs(Xx) ** 2
    plt.figure()
    plt.title('Spectrogram')
    plt.pcolormesh(T, F, abs(Xx.T))

    # Horz & Vert Median Filtered
    print('Horz Median Filtering')
    Yh = HorzMedianFilter(Y, 11)
    print('Vert Median Filtering')
    Yp = VertMedianFilter(Y, 101)
    plt.figure()
    plt.subplot(211)
    plt.title('Horizontal Median Filter')
    plt.pcolormesh(T, F, Yh.T)
    plt.subplot(212)
    plt.title('Vertical Median Filter')
    plt.pcolormesh(T, F, Yp.T)

    #Masking
    ##Binary
    print('Generating Binary Masks')
    Mhb, Mpb = BinaryMask(Yh, Yp)
    plt.figure()
    plt.subplot(211)
    plt.title('Horizontal Binary Mask')
    plt.pcolormesh(T,F,Mhb.T)
    plt.subplot(212)
    plt.title('Vertical Binary Mask')
    plt.pcolormesh(T,F,Mpb.T)

    ##Soft
    print('Generating Soft Masks')
    Mhs, Mps = SoftMask(Yh, Yp)
    plt.figure()
    plt.subplot(211)
    plt.title('Horizontal Soft Mask')
    plt.pcolormesh(T,F,Mhs.T)
    plt.subplot(212)
    plt.title('Vertical Soft Mask')
    plt.pcolormesh(T,F,Mps.T)


    #Apply
    print('Applying Soft Mask')
    Xh = Mhs * Xx
    Xp = Mps * Xx
    plt.figure()
    plt.subplot(211)
    plt.title('Harmonic Spectrogram')
    plt.pcolormesh(T,F,abs(Xh.T))
    plt.subplot(212)
    plt.title('Percussive Spectrogram')
    plt.pcolormesh(T,F,abs(Xp.T))

    # Get audio signals
    print('Reconstructing Signals')
    th, xh = Spectral.iSTFT(Xh, Fs, 2**13, 2**12)
    tp, xp = Spectral.iSTFT(Xp, Fs, 2**13, 2**12)
    plt.figure()
    plt.subplot(211)
    plt.title('Reconstructed Harmonic Audio Signal')
    plt.plot(th,xh)
    plt.subplot(212)
    plt.title('Reconstructed Percussive Audio Signal')
    plt.plot(tp,xp)
    
    
    

    
    

    input('press return to continue')

