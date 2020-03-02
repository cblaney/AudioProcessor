import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import Spectral
import HarmonicPercussiveSeparation as HPS

from matplotlib import interactive
interactive(True)

fs = 10e3
N = 10e4
t = np.arange(N)/fs

# CW Signals
# - s1: Pure CW
a1 = 1
f1 = (fs/2) / 16
s1 = fs*a1 * np.exp(-2j*np.pi*f1*t)

# - s2: fmod CW
a2 = 2
f2_c = (fs/2)*2 / 16
f2_m = N/fs / 4
s2_m = np.cos(2*np.pi*f2_m*t)
s2 = fs*a2 * np.exp(-2j*np.pi*(f2_c*t + 4*s2_m))

# - pulsed signal
a3 = 100
f3 = (fs/2)*3 / 16
pri = 0.5
pw = 0.0005
t3 = np.array((np.modf(t/pri)[0]*pri < pw), dtype=int) * t
s3_c = fs*a3 * np.exp(-2j*np.pi*f3*t3)
s3_w = np.array((np.modf(t/pri)[0]*pri < pw), dtype=int)
s3 = s3_c * s3_w

# - noise
n = np.random.normal(0,2*fs, len(t))

# - add'em up
x = s3 + s2 + s1 + n


plt.figure()
plt.title('Complex Signal')
plt.subplot(211)
plt.title('Real')
plt.plot(t,np.real(x))
plt.subplot(212)
plt.title('Abs')
plt.plot(t,np.abs(x))

# Take STFT
N_fft = 2**9
T, F, Xx = Spectral.STFT(np.real(x), fs, N_fft, N_fft, real=True)
plt.figure()
plt.title('Spectrogram')
plt.pcolormesh(T, F, abs(Xx.T))

Y = (abs(Xx) ** 2)

# Filter
print('Horz Median Filtering')
Yh = HPS.HorzMedianFilter(Y, 3)
print('Vert Median Filtering')
Yp = HPS.VertMedianFilter(Y, int(N_fft/8))
plt.figure()
plt.subplot(211)
plt.title('Horizontal Median Filter')
plt.pcolormesh(T, F, Yh.T)
plt.subplot(212)
plt.title('Vertical Median Filter')
plt.pcolormesh(T, F, Yp.T)

# Mask
##Binary
print('Generating Binary Masks')
Mhb, Mpb = HPS.BinaryMask(Yh, Yp)
plt.figure()
plt.subplot(211)
plt.title('Horizontal Binary Mask')
plt.pcolormesh(T,F,Mhb.T)
plt.subplot(212)
plt.title('Vertical Binary Mask')
plt.pcolormesh(T,F,Mpb.T)

##Soft
print('Generating Soft Masks')
Mhs, Mps = HPS.SoftMask(Yh, Yp)
plt.figure()
plt.subplot(211)
plt.title('Horizontal Soft Mask')
plt.pcolormesh(T,F,Mhs.T)
plt.subplot(212)
plt.title('Vertical Soft Mask')
plt.pcolormesh(T,F,Mps.T)

#Apply
print('Applying Soft Mask')
Xhs = Mhs * Xx
Xps = Mps * Xx
plt.figure()
plt.subplot(211)
plt.title('Soft Horizontal Spectrogram')
plt.pcolormesh(T,F,abs(Xhs.T))
plt.subplot(212)
plt.title('Soft Vertical Spectrogram')
plt.pcolormesh(T,F,abs(Xps.T))

#Apply
print('Applying Binary Mask')
Xhb = Mhb * Xx
Xpb = Mpb * Xx
plt.figure()
plt.subplot(211)
plt.title('Binary Horizontal Spectrogram')
plt.pcolormesh(T,F,abs(Xhb.T))
plt.subplot(212)
plt.title('Binary Vertical Spectrogram')
plt.pcolormesh(T,F,abs(Xpb.T))


input('press return to continue')

