import numpy as np


def highpassDesign(sampling_rate,cutoff_frequencies,Frequency_Resolution):
    #Figure out the ideal frequency response of high pass filter
    fs = sampling_rate
    M = int(fs/Frequency_Resolution)
    k = int((cutoff_frequencies/fs) *M)
    X = np.ones(M)
    X[0:k + 1] = 0
    X[M - k:M + 1] = 0

    #Do the Inverse Fourier Transformation and Get the real part
    x = np.fft.ifft(X)
    x = np.real(x)

    #make a shift for response in time domain and Add window function
    h = np.zeros(M)
    h[0:int(M/2)] = x[int(M/2):M]
    h[int(M/2):M] = x[0:int(M/2)]

    h = h * np.hanning(M)

    return h

def bandstopDesign(sampling_rate,cutoff_frequencies,Frequency_Resolution):
        fs = sampling_rate
        M = int(fs / Frequency_Resolution)
        k1 = int(cutoff_frequencies[0]/fs * M)
        k2 = int(cutoff_frequencies[1] / fs * M)
        X = np.ones(M)
        X[k1:k2+1] = 0
        X[M-k2:M-k1+1] = 0

        # Do the Inverse Fourier Transformation and Get the real part
        x = np.fft.ifft(X)
        x = np.real(x)

        # make a shift for response in time domain and Add window function
        h = np.zeros(M)
        h[0:int(M / 2)] = x[int(M / 2):M]
        h[int(M / 2):M] = x[0:int(M / 2)]

        h = h * np.hanning(M)

        return h

class FIRfilter:
    def __init__(self,_coefficients,Frequency_Resolution = 1):
        self.ntaps = len(_coefficients)  # ntaps is the number of coefficients
        self.coefficients = _coefficients  # define the variable called coefficients to save the coeffient of FIR
        self.buffer = np.zeros(self.ntaps)  # buffer is used to save the input value
        self.Frequency_Resolution = Frequency_Resolution


    def dofilter(self,v):
        output =0
        for j in range(self.ntaps - 1):
            self.buffer[self.ntaps - 1 - j] = self.buffer[self.ntaps - 2 - j]  # as time goes by move the past value x(n) to x(n-1)
        self.buffer[0] = v  # assign the v to buffer[0] as the current input value

        for i in range(len(self.coefficients)):
            output += self.coefficients[i]*self.buffer[i]

        #return np.inner(self.buffer, self.coefficients)
        return output

    def doFilterAdaptive(self, signal, noise, learningRate):
        cancellor = self.dofilter(noise)
        error = signal - cancellor
        for j in range(self.ntaps):
            self.coefficients[j] = self.coefficients[j] + error * learningRate * self.buffer[j]

        return error