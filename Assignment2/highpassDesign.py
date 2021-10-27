import numpy as np

def highpassDesign(Fs, cutoff, del_F = 1):

    #Select number of taps using the formula for resolution. Default resolution is 1
    M = Fs / del_F

    #Multiply Normalized frequency with Sampling Rate
    k = int(cutoff/M * Fs)

    #Creating Ideal frequency response
    x = np.ones(M)

    x[0:k] = 0
    x[M-k:M-1] = 0

    #Perform Inverse fourier transform to get impulse response 
    np.fft.ifft(x)

    #convert complex values to real values
    np.real(x)
    h = np.zeros(M)

    #Shift System
    h[0:int(M/2)] = x[int(M/2):M]
    h[int(M/2):M] = x[0:int(M/2)]

    #Imporve system performance using Hamming window
    h = h  * np.hamming(M)

    return h





