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



# Testing code
if(__name__ == "__main__"):
    # Testing tools
    from matplotlib import interactive
    interactive(True)

    # Local tools
    import WavReader
    import Spectral

    # Collect mono audio
    wr = WavReader.WavReader('./../audio/C Chord - 1.3 - Acoustic Piano.wav')
    x = wr.getMono()
    plt.figure()
    plt.plot(x)

    # Power Spectrogram
    print('Calculating Spectrogram')
    T, F, Xx = Spectral.STFT(x, wr.sampleRate, 2**13, 2**12)
    Y = abs(Xx)
    #Y = abs(Xx) ** 2
    plt.figure()
    plt.pcolormesh(T, F, abs(Y.T))

    # Horz & Vert Median Filtered
    print('Horz Median Filtering')
    Yh = HorzMedianFilter(Y, 11)
    print('Vert Median Filtering')
    Yp = VertMedianFilter(Y, 11)

    plt.figure()
    plt.pcolormesh(T, F, abs(Yh.T))
    plt.figure()
    plt.pcolormesh(T, F, abs(Yp.T))


    input('press return to continue')

