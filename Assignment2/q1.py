
# Create two functions which calculate and return the FIR filter coefficients
# numerically (= using python’s IFFT command) for a
# a) highpass filter and
# b) a bandstop filter.
# Name these functions “highpassDesign” and “bandstopDesign”.
# Both functions should automatically decide how many coefficients are
# required. The function arguments should be the sampling rate and the
# cutoff frequencies (and any other optional arguments you like to
# provide). Feel free to put both functions in a class.

import numpy as np
import pylab as pl
import scipy as sp

def highpassDesign(sampling_rate,cutoff_frequencies,Frequency_Resolution):
    fs = sampling_rate
    M = int(fs/Frequency_Resolution)
    k = int(cutoff_frequencies/fs *M)
    X = np.ones(M)
    X[0:k] = 0
    X[M - k:M - 1] = 0
    return X




def bandstopDesign(sampling_rate,cutoff_frequencies,Frequency_Resolution):
        fs = sampling_rate
        M = fs / Frequency_Resolution
        k1 = int(cutoff_frequencies[0]/fs * M)
        k2 = int(cutoff_frequencies[1] / fs * M)
        X = np.ones(M)
        X[k1:k2+1] = 0
        X[M-k2:M-k1+1] = 0
        X = np.real(X)
        return X

if __name__ == '__main__':
    print(highpassDesign(250, 50, 1))
    print(bandstopDesign(250,[45,55],1))