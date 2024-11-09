import numpy as np
from numpy import pi, exp, log10
from numpy.fft import fft, fftshift, fftfreq, ifft

import scipy
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

SPEC_M = 4096;
FREQ_BINS = 34
N_SPEC = 2049
FS = 44100
N_SAMPLES = 1321967

def load_wav(fn):
    fs, x = scipy.io.wavfile.read(fn)
    x = np.array(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError('Input wav file must be stereo')
    if fs != FS:
        raise ValueError('Input wav file must have sample rate of 44100')
    if (x.shape[0] != N_SAMPLES) or (x.shape[1] != 2):
        raise ValueError('Input wav file must have 1321967 samples')
    l = x[:, 0]
    r = x[:, 1]
    return fs, l, r

def normalize_audio(l, r):
    m = max(np.linalg.norm(l), np.linalg.norm(r))
    l = l / m
    r = r / m
    return l, r, m

def file_to_feature_label(f, stft):
    fs, l, r = load_wav(f)

    # normalize audio
    l, r, m = normalize_audio(l, r)

    # create mono track
    s = (l + r) / 2
    BS = b_matrix() @ spectrogram(stft, s)

    ps = np.load(f + '_ps.npz')
    P_IID, P_IC = ps['P_IID'], ps['P_IC']
    P = parameters_concat(P_IID, P_IC)
    
    # s - the mono track
    # BS - the binned spectrogram of the mono track (feature)
    # P_IID - the IID parameters
    # P_IC - the IC parameters
    # P - the parameters concatenated
    # m - the normalization factor (needed for decoding)
    return s, BS, P_IID, P_IC, P, m

def get_stft(fs):
    M = SPEC_M # frame size
    overlap = 0.75 # hann window overlap (used to calculate hop size)
    hop = int(M*(1-overlap)) # hop size

    window = hann(M)

    stft = ShortTimeFFT(win=window, hop=hop, fs=fs)
    return stft

# convert index between 0 and N_SPEC to 0 and FREQ_BINS
def k_to_b(k):
    return np.round((FREQ_BINS-1) * k / (N_SPEC-1)).astype(int)

def b_matrix():
    B = np.zeros((FREQ_BINS, N_SPEC), dtype=int)
    for k in range(N_SPEC):
        B[k_to_b(k), k] = 1
    return B

def parameters_concat(P_IID, P_IC):
    return np.concatenate((P_IID, P_IC), axis=0)

def parameters_split(P):
    P_IID = P[0:FREQ_BINS]
    P_IC = P[FREQ_BINS:]
    return P_IID, P_IC

def complex_to_real(x):
    return np.concatenate((x.real, x.imag), axis=0)

def real_to_complex(x):
    N = len(x) // 2
    return x[0:N] + 1j*x[N:]

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

    # avoid division by zero
    e = 1e-10
    p_RR[p_RR == 0.] = e
    p_LL[p_LL == 0.] = e

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


    Y_mixed = np.zeros((2, N_SPEC, N1), dtype=complex)

    c_1 = np.sqrt(2*c**2 / (1 + c**2))
    c_2 = np.sqrt(2 / (1 + c**2))
    mu = 0.5*np.arccos(P_IC)
    nu = mu*(c_2 - c_1) / np.sqrt(2)

    R_A = np.zeros((FREQ_BINS, 2, 2, N1), dtype=complex)
    R_A[:, 0, 0] = c_1*np.cos(nu + mu)
    R_A[:, 0, 1] = c_1*np.sin(nu + mu)
    R_A[:, 1, 0] = c_2*np.cos(nu - mu)
    R_A[:, 1, 1] = c_2*np.sin(nu - mu)

    for k in np.arange(0, N_SPEC):
        b = k_to_b(k)
        
        Y_mixed[0][k] = R_A[b][0][0]*S[k] + R_A[b][0][1]*S_d[k]
        Y_mixed[1][k] = R_A[b][1][0]*S[k] + R_A[b][1][1]*S_d[k]
    
    y_mixed = np.array([inverse_spectrogram(stft, Y_mixed[0]), inverse_spectrogram(stft, Y_mixed[1])])
    return y_mixed