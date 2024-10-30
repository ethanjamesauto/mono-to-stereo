import numpy as np
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift, fftfreq, ifft

import scipy

# convert index between 0 and 2049 to 0 and 34
def k_to_b(k):
    return np.round(33 * k / 2048).astype(int)

def b_matrix():
    B = np.zeros((34, 2049), dtype=int)
    for k in range(2049):
        B[k_to_b(k), k] = 1
    return B

# of 1s in each row of B
def b_matrix_counts():
    return np.sum(b_matrix(), axis=1)

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

def schroeder_phase_complex():
    Ns = 640
    n = np.arange(Ns)
    h = np.zeros(Ns)
    for k in range(0, Ns//2):
        h += (2/Ns) * np.cos(2*pi*k*n/Ns + 2*pi*k*(k-1)/Ns)
    return h

def decorrelate(x):
    y = scipy.signal.fftconvolve(x, schroeder_phase_complex()) # should be much faster than np.convolve
    return y

def decode(stft, s, P_IID, P_IC):
    N = len(s)
    s_d = decorrelate(s)[0:N]
    # scipy.io.wavfile.write('freebird_decorrelated.wav', fs, s_d.astype(np.int16))

    S = spectrogram(stft=stft, x=s)
    S_d = spectrogram(stft=stft, x=s_d)

    c = 10**(P_IID/20) # TODO: the paper says /20, but I think it might be /10
    N1 = len(c[0])

    n_spec = 2049

    Y_mixed = np.zeros((2, n_spec, N1), dtype=complex)

    c_1 = np.sqrt(2*c**2 / (1 + c**2))
    c_2 = np.sqrt(2 / (1 + c**2))
    mu = 0.5*np.arccos(P_IC)
    nu = mu*(c_2 - c_1) / np.sqrt(2)

    R_A = np.zeros((34, 2, 2, N1), dtype=complex)
    R_A[:, 0, 0] = c_1*np.cos(nu + mu)
    R_A[:, 0, 1] = c_1*np.sin(nu + mu)
    R_A[:, 1, 0] = c_2*np.cos(nu - mu)
    R_A[:, 1, 1] = c_2*np.sin(nu - mu)

    for k in np.arange(0, n_spec):
        b = k_to_b(k)
        
        Y_mixed[0][k] = R_A[b][0][0]*S[k] + R_A[b][0][1]*S_d[k]
        Y_mixed[1][k] = R_A[b][1][0]*S[k] + R_A[b][1][1]*S_d[k]
    
    y_mixed = np.array([inverse_spectrogram(stft, Y_mixed[0]), inverse_spectrogram(stft, Y_mixed[1])])
    return y_mixed