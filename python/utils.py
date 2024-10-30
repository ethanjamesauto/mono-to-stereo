import numpy as np
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift, fftfreq, ifft

def b_matrix():
    B = np.zeros((34, 2049))
    r = np.round(np.linspace(0, 33, 2049)).astype(int)
    for i in range(34):
        B[i, r == i] = 1
    return B

def spectrogram(stft, x):
    X = stft.stft(x)
    return X

def inverse_spectrogram(stft, X):
    x = stft.istft(X)
    return x

def cross_spectrogram(X, Y):
    B = b_matrix()
    X1 = B @ (X * Y.conj())
    return X1

def encode(stft, l, r):
    L = spectrogram(stft=stft, x=l)
    R = spectrogram(stft=stft, x=r)

    N = len(l)

    p_LL = cross_spectrogram(L, L)
    p_RR = cross_spectrogram(R, R)
    p_LR = cross_spectrogram(L, R)

    P_IID = 10*np.log10(p_LL / p_RR)
    P_IC = p_LR.real / np.sqrt(p_LL * p_RR)
    return P_IID, P_IC
